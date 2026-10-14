"""
This code was generated with the help of ChatGPT 5.6 Sol but was checked and validated.

Ground-truth mean survival for TransformerLSR simulation data.

Example (from the Notebook):
    python oracle_survival_variant2.py \
        --data_path data/DIVAT_sim_missing_100_visit_1000_long_3_seed_3.pkl \
        --patient_info_path data/DIVAT_sim_missing_100_visit_1000_long_3_seed_3_patientInfo.pkl \
        --simulation_py data_simulation.py \
        --n_paths 500 \
        --horizon 16000 \
        --output_path results/oracle_survival_seed_3.pkl
"""

# Imports
import argparse
import copy
import importlib.util
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Load the data simulation
def load_divat_env(simulation_py):
    simulation_py = Path(simulation_py).resolve()
    spec = importlib.util.spec_from_file_location("transformerlsr_data_simulation", simulation_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DIVAT_env


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_split_and_landmark(data, train_frac=0.6, val_frac=0.2, lt_quantile=0.1):
    # Using same split
    data = data.loc[data["obstime"] <= data["time"]].copy()

    ids = np.sort(data["id"].unique())
    n = len(ids)

    n_train = int(train_frac * n)
    n_val_end = int((train_frac + val_frac) * n)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_val_end]
    test_ids = ids[n_val_end:]

    train_subjects = (
        data.loc[data["id"].isin(train_ids), ["id", "time"]]
        .drop_duplicates(subset="id")
        .sort_values("id")
    )
    LT = float(np.quantile(train_subjects["time"].to_numpy(), lt_quantile))

    test_subjects = (
        data.loc[data["id"].isin(test_ids), ["id", "time"]]
        .drop_duplicates(subset="id")
        .sort_values("id")
    )

    eligible_ids = test_subjects.loc[test_subjects["time"] > LT, "id"].to_numpy()

    # Keep only subjects that actually have at least one recorded visit <= LT.
    eligible_ids = np.array(
        [
            pid for pid in eligible_ids
            if np.any((data["id"] == pid) & (data["obstime"] <= LT))
        ]
    )

    return data, train_ids, val_ids, test_ids, eligible_ids, LT


def get_last_visit_before_landmark(data, patient_id, LT):
    rows = (
        data.loc[(data["id"] == patient_id) & (data["obstime"] <= LT)]
        .sort_values(["obstime", "visit"])
    )
    if len(rows) == 0:
        raise ValueError(f"Patient {patient_id}: no visit at or before LT={LT:.6f}.")
    return rows.iloc[-1]


# We initialise at last visit to draw MC samples for a sufficient ground truth 
def initialise_env_at_last_visit(DIVAT_env, patient_info, last_row, data, timeout=1000):
    env = DIVAT_env(timeout=timeout)

    # Create starting point at the last visit before LT
    # and restore all patient-specific information
    env.b_il = np.asarray(patient_info["b_il"]).copy()
    env.b_il_tl = np.asarray(patient_info["b_il_tl"]).copy()
    env.BMI = patient_info["BMI"]
    env.ageD = patient_info["ageD"]
    env.DGF = patient_info["DGF"]

    env.time = float(last_row["obstime"])
    env.steps_elapsed = int(last_row["visit"]) if "visit" in last_row.index else 0
    env.tox = (
        float(last_row["tox"])
        if ("tox" in last_row.index and pd.notna(last_row["tox"]))
        else 0.0
    )

    patient_id = int(last_row["id"])
    t_last = float(last_row["obstime"])

    # Helper: Get the last available value of a variable before or at the current visit
    def get_last_available_value(variable):
        values = (
            data.loc[
                (data["id"] == patient_id)
                & (data["obstime"] <= t_last)
                & (data[variable].notna()),
                ["obstime", variable]
            ]
            .sort_values("obstime")
        )

        if len(values) == 0:
            return np.nan

        return values.iloc[-1][variable]

    # Use value at last visit; if missing, use last available value (LOCF)
    y1 = last_row.get("Y1", np.nan)
    if pd.isna(y1):
        y1 = get_last_available_value("Y1")

    y2 = last_row.get("Y2", np.nan)
    if pd.isna(y2):
        y2 = get_last_available_value("Y2")

    y3 = last_row.get("Y3", np.nan)
    if pd.isna(y3):
        y3 = get_last_available_value("Y3")

    # Check whether a value could be found for all variables
    if pd.isna(y1) or pd.isna(y2) or pd.isna(y3):
        raise ValueError(
            f"Patient {patient_id}: no previous observed value available "
            f"for Y1, Y2, or Y3."
        )

    # Restore longitudinal state
    env.tl = np.array([float(y1)])
    env.y = np.array([float(y2)])

    # Treatment active after the last visit
    current_dose = float(y3)

    # Disable censoring for oracle survival generation
    env.censortime = np.inf

    return env, current_dose

# Draw visits
def sample_first_visit_conditional_on_no_visit_until_LT(env, current_dose, LT, max_tries=1000):
    if env.time > LT:
        raise ValueError("Environment start time is after LT.")

    for _ in range(max_tries):
        accepted, t_accept = env.sample_visit(current_dose)
        if accepted:
            next_visit = float(t_accept)
        else:
            # Matches sample_treatment(): if thinning returns no accepted visit,
            # the original DGP advances by max_visit.
            next_visit = float(env.time + env.max_visit)

        if next_visit > LT:
            return next_visit

    raise RuntimeError(
        f"Could not sample a visit after LT={LT:.6f} in {max_tries} tries."
    )



def fill_hazard_segment(env, dose, segment_left, segment_right, plot_times, cum_hazard, base_hazard):
    """
    Fill cumulative hazard values for plot_times in [segment_left, segment_right].
    env.time must remain the visit time anchoring hazard_fun for this interval.
    """
    if segment_right <= segment_left:
        return base_hazard

    mask = (plot_times >= segment_left) & (plot_times <= segment_right)
    idx = np.where(mask)[0]

    for k in idx:
        t = float(plot_times[k])
        inc = env.cumulative_prob(t_upper=t, t_lower=segment_left, di=dose)
        cum_hazard[k] = base_hazard + float(inc)

    total_inc = env.cumulative_prob(
        t_upper=float(segment_right),
        t_lower=float(segment_left),
        di=dose,
    )
    return base_hazard + float(total_inc)


def simulate_one_oracle_path(
    DIVAT_env,
    patient_info,
    last_row,
    LT,
    plot_times,
    timeout=1000,
):
    """
    Simulate one future longitudinal/visit/treatment path and return the
    conditional survival curve from LT onward, based on the true DGP hazard.

    Death is not sampled. Survival is obtained as exp(- cumulative true hazard).
    """
    env, current_dose = initialise_env_at_last_visit(
        DIVAT_env, patient_info, last_row, timeout=timeout
    )

    horizon = float(plot_times[-1])
    cum_hazard = np.full(len(plot_times), np.nan, dtype=float)
    base_hazard = 0.0

    # The known history tells us that no visit occurred before LT.
    next_visit = sample_first_visit_conditional_on_no_visit_until_LT(
        env, current_dose, LT
    )

    # Initial interval: last observed visit -> first future visit.
    # Survival is conditioned on being alive at LT, so integration starts at LT.
    segment_left = float(LT)
    segment_right = min(float(next_visit), horizon)

    base_hazard = fill_hazard_segment(
        env,
        current_dose,
        segment_left,
        segment_right,
        plot_times,
        cum_hazard,
        base_hazard,
    )

    if next_visit >= horizon:
        return np.exp(-cum_hazard)

    # Arrive at the first future visit: generate fresh Y1/Y2 under the old dose.
    env.update_long(current_dose, float(next_visit), update_obs=True)
    env.tox = env.toxicity(float(next_visit), env.time, current_dose)
    env.time = float(next_visit)

    # Continue with the original DGP. At each visit a new dose is drawn from
    # the newly generated current Y2, and then the next visit time is sampled.
    safety_counter = 0
    while env.time < horizon:
        safety_counter += 1
        if safety_counter > timeout + 10000:
            raise RuntimeError("Safety stop: too many future visits.")

        new_dose, delta_t, _ = env.sample_treatment()
        new_dose = float(new_dose)
        next_visit = float(env.time + delta_t)

        segment_left = float(env.time)
        segment_right = min(next_visit, horizon)

        base_hazard = fill_hazard_segment(
            env,
            new_dose,
            segment_left,
            segment_right,
            plot_times,
            cum_hazard,
            base_hazard,
        )

        if next_visit >= horizon:
            break

        # Generate the longitudinal observation at the next visit.
        env.update_long(new_dose, next_visit, update_obs=True)
        env.tox = env.toxicity(next_visit, env.time, new_dose)
        env.time = next_visit
        current_dose = new_dose

    if np.isnan(cum_hazard).any():
        missing_idx = np.where(np.isnan(cum_hazard))[0]
        raise RuntimeError(
            f"Some plot times were not filled: indices {missing_idx[:10].tolist()}"
        )

    return np.exp(-cum_hazard)


def compute_oracle_survival(
    data,
    patient_info,
    DIVAT_env,
    n_paths=100,
    horizon=16000.0,
    n_plot=100,
    train_frac=0.6,
    val_frac=0.2,
    lt_quantile=0.1,
    timeout=1000,
    mc_seed=20261014,
    verbose=True,
):
    data, train_ids, val_ids, test_ids, eligible_ids, LT = get_split_and_landmark(
        data,
        train_frac=train_frac,
        val_frac=val_frac,
        lt_quantile=lt_quantile,
    )

    # Include LT itself explicitly so all curves start at 1.
    plot_times = np.linspace(LT, horizon, n_plot)

    rng = np.random.default_rng(mc_seed)
    random.seed(mc_seed)
    np.random.seed(mc_seed)

    patient_curves = {}
    start_rows = {}

    for j, pid in enumerate(eligible_ids, start=1):
        pid = int(pid)
        last_row = get_last_visit_before_landmark(data, pid, LT)
        start_rows[pid] = {
            "last_visit_time": float(last_row["obstime"]),
            "Y1": None if pd.isna(last_row.get("Y1", np.nan)) else float(last_row["Y1"]),
            "Y2": None if pd.isna(last_row.get("Y2", np.nan)) else float(last_row["Y2"]),
            "Y3": None if pd.isna(last_row.get("Y3", np.nan)) else float(last_row["Y3"]),
        }

        if verbose:
            print(
                f"[{j:02d}/{len(eligible_ids):02d}] patient {pid}: "
                f"last visit={last_row['obstime']:.3f}, LT={LT:.3f}, "
                f"Y3={'NaN' if pd.isna(last_row['Y3']) else f'{last_row['Y3']:.3f}'}"
            )

        # Use a deterministic patient-specific seed so results are reproducible.
        patient_seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(patient_seed)
        random.seed(patient_seed)

        curves = np.empty((n_paths, len(plot_times)), dtype=float)

        for m in range(n_paths):
            curves[m, :] = simulate_one_oracle_path(
                DIVAT_env=DIVAT_env,
                patient_info=patient_info[pid],
                last_row=last_row,
                LT=LT,
                plot_times=plot_times,
                timeout=timeout,
            )

        patient_curves[pid] = curves.mean(axis=0)

    oracle_by_patient = np.vstack([patient_curves[int(pid)] for pid in eligible_ids])
    mean_oracle_surv = oracle_by_patient.mean(axis=0)

    return {
        "LT": LT,
        "plot_times": plot_times,
        "eligible_test_ids": np.asarray(eligible_ids, dtype=int),
        "test_ids": np.asarray(test_ids, dtype=int),
        "n_paths": int(n_paths),
        "horizon": float(horizon),
        "oracle_surv_by_patient": oracle_by_patient,
        "mean_oracle_surv": mean_oracle_surv,
        "start_rows": start_rows,
        "mc_seed": int(mc_seed),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--patient_info_path", required=True, type=str)
    parser.add_argument("--simulation_py", default="data_simulation.py", type=str)
    parser.add_argument("--output_path", required=True, type=str)

    parser.add_argument("--n_paths", default=500, type=int)
    parser.add_argument("--horizon", default=16000.0, type=float)
    parser.add_argument("--n_plot", default=100, type=int)
    parser.add_argument("--timeout", default=1000, type=int)
    parser.add_argument("--mc_seed", default=20261014, type=int)

    parser.add_argument("--train_frac", default=0.6, type=float)
    parser.add_argument("--val_frac", default=0.2, type=float)
    parser.add_argument("--lt_quantile", default=0.1, type=float)
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    data = load_pickle(args.data_path)
    patient_info = load_pickle(args.patient_info_path)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data_path must contain a pandas DataFrame.")
    if len(patient_info) < data["id"].nunique():
        raise ValueError(
            "patientInfo has fewer entries than the number of patients in data."
        )

    DIVAT_env = load_divat_env(args.simulation_py)

    result = compute_oracle_survival(
        data=data,
        patient_info=patient_info,
        DIVAT_env=DIVAT_env,
        n_paths=args.n_paths,
        horizon=args.horizon,
        n_plot=args.n_plot,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        lt_quantile=args.lt_quantile,
        timeout=args.timeout,
        mc_seed=args.mc_seed,
        verbose=not args.quiet,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print("\nDone.")
    print(f"LT: {result['LT']:.6f}")
    print(f"Eligible test patients: {len(result['eligible_test_ids'])}")
    print(f"IDs: {result['eligible_test_ids'].tolist()}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

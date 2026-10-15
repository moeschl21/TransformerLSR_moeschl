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
def initialise_env_at_last_visit(DIVAT_env, patient_info, last_row, timeout=1000):
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
    
    # Oracle dataset is complete, so use the true simulated values
    # at the last visit before LT directly.
    y1 = last_row["Y1"]
    y2 = last_row["Y2"]
    y3 = last_row["Y3"]
    
    if pd.isna(y1) or pd.isna(y2) or pd.isna(y3):
        raise ValueError(
            f"Patient {patient_id}: unexpected missing value "
            f"in complete oracle dataset."
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

def simulate_one_oracle_path(
    DIVAT_env,
    patient_info,
    last_row,
    LT,
    horizon=16000.0,
    timeout=1000,
):
    # Initialise patient at the last visit before LT
    env, current_dose = initialise_env_at_last_visit(
        DIVAT_env,
        patient_info,
        last_row,
        timeout=timeout,
    )

    # Prediction horizon is the administrative end point
    env.censortime = float(horizon)

    # Avoid stopping because of the original visit-count timeout
    env.timeout = 10**9

    # Draw first future visit conditional on no visit before LT
    next_visit = sample_first_visit_conditional_on_no_visit_until_LT(
        env,
        current_dose,
        LT,
    )

    # Never simulate beyond the prediction horizon
    step_end = min(next_visit, horizon)
    delta_t = step_end - env.time

    _, _, done, death, _, _ = env.step(
        [current_dose, delta_t]
    )

    # Path incompatible with known survival beyond LT
    if death and env.time <= LT:
        return None

    # Death after LT
    if death:
        return float(env.time)

    # Survived to prediction horizon
    if done or env.time >= horizon:
        return np.inf

    # Continue future simulation
    while env.time < horizon:

        new_dose, delta_t, _ = env.sample_treatment()

        # Do not simulate beyond horizon
        delta_t = min(delta_t, horizon - env.time)

        _, _, done, death, _, _ = env.step(
            [new_dose, delta_t]
        )

        if death:
            return float(env.time)

        if done or env.time >= horizon:
            return np.inf

    return np.inf


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
        
        death_times = []
        
        while len(death_times) < n_paths:
        
            death_time = simulate_one_oracle_path(
                DIVAT_env=DIVAT_env,
                patient_info=patient_info[pid],
                last_row=last_row,
                LT=LT,
                horizon=horizon,
                timeout=timeout,
            )
        
            # Path is incompatible with the known condition T > LT.
            # Redraw the Monte Carlo path, not the patient.
            if death_time is None:
                continue
        
            death_times.append(death_time)
        
        death_times = np.asarray(death_times, dtype=float)
        
        patient_curves[pid] = np.array([
            np.mean(death_times > t)
            for t in plot_times
        ])

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

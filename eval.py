import torch
import torch.nn as nn
import argparse
import logging
import time
import os
import random
import pickle
import copy
from TransformerLSR import TransformerLSR

from functions import (get_tensors,get_tensors_likelihood)
from brier import brier, brier_fast

from lifelines import KaplanMeierFitter
# Other Python libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

# JM Ignore future warnings for better debugging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# computes the sum of square errors for event likelihood
# JM Lam und Zeta ist die Integralapproximation pro Intervall
# JM Name irrefürhend hier wird nur squared Error berechnet nicht MSE!
def MSE_likelihood(visit_inten,Lam,surv_inten,Zeta,batch):
    long_mask = batch["longmask"]
    batch_mask = batch["mask"]
    death_mask = batch["intenmask"]
    
    #intensity is from the second visit to the last visit (t_0 as starting point)
    visit_event_ll = (torch.log(visit_inten)*long_mask[:,1:]).sum(dim=-1) # Σ log λ(t_j)
    visit_non_ll = (Lam * batch_mask).sum(dim=-1) # ∫ λ(t)dt und dann aufsummiert
    visit_pred_likelihood = visit_event_ll - visit_non_ll
    visit_truth_likelihood = batch["visit_ll"] # Ground Truth
    visit_se = torch.sum((visit_pred_likelihood-visit_truth_likelihood)**2) # Sum of SE per Batch
    visit_se_out = visit_se.cpu().numpy()
    
    # survival error computation (Similar as above)
    surv_event_ll = (torch.log(surv_inten)*death_mask[:,1:]).sum(dim=-1) # log h(T)
    surv_non_ll = (Zeta * batch_mask).sum(dim=-1) # ∫ h(t)dt
    surv_pred_likelihood = surv_event_ll - surv_non_ll
    surv_truth_likelihood = batch["surv_ll"]
    surv_se = torch.sum((surv_pred_likelihood-surv_truth_likelihood)**2) # Sum of SE per Batch
    surv_se_out = surv_se.cpu().numpy()
    return visit_se_out,surv_se_out # Sum of SE per Batch

# JM Fix DeprecationWarning np.trapz to np.trapezoid
def get_integrated(x, times):
    return np.trapezoid(x,times) / (max(times)-min(times))


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)              
    parser.add_argument("--batch_size", default=16, type=int)      # Mini batch size for networks
    parser.add_argument("--num_enc_layer", default=4, type=int)  
    parser.add_argument("--num_dec_layer", default=4, type=int)   
    parser.add_argument("--num_head", default=4, type=int)      
    parser.add_argument("--model_size", default=32, type=int)    # JM Embedding   
    parser.add_argument('--suffix', type=str, default='eval')
    parser.add_argument('--model', type=str, default='LSR')
    parser.add_argument("--d_long", default=3, type=int) 
    parser.add_argument('--data', type=str, default='DIVAT_sim_1000_visit_1000_long_3')
    parser.add_argument("--local", action="store_true")   # local test mode
    parser.add_argument("--Y1_missing", default=0, type=float)
    parser.add_argument("--Y2_missing", default=0, type=float)
    parser.add_argument("--Y3_missing", default=0, type=float)
    parser.add_argument("--inten_weight", default=0.01, type=float) # JM Loss weights
    parser.add_argument("--surv_weight", default=0.1, type=float)
    parser.add_argument("--lr", default=0.0003, type=float) # learning rate
    args = parser.parse_args()


    
    # make logger here JM Eig wie print
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                datefmt="%a %b %d %H:%M:%S %Y") # JM Thats how the print will look like then
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)
    work_dir = os.path.join('./work_dir',
                                time.strftime("%Y-%m-%d", time.localtime())) # JM Ordner für die loggs
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    time_prefix = time.strftime("%H:%M:%S", time.localtime())
    full_path = work_dir + '/'+time_prefix +'_'+args.data+'_'+args.model+'_'+"head_"+str(args.num_head)+'_'+ \
                                        "enc_layer_"+str(args.num_enc_layer)+'_'+"dec_layer_"+str(args.num_dec_layer)+'_'+"size_"+str(args.model_size)+'_'+ \
                                        '_'+"visit_weight_"+str(args.inten_weight)+'_'+"surv_weight_"+str(args.surv_weight)+ \
                                        '_'+"lr_"+str(args.lr)+"Y1miss_"+str(args.Y1_missing)+"Y2miss_"+str(args.Y2_missing)+"Y3miss_"+str(args.Y3_missing)+args.suffix +'-log.txt'
    if not args.local:
        fHandler = logging.FileHandler(full_path, mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)
        logger.addHandler(fHandler)


    # log meta-data
    logger.info(args)


    if not os.path.exists("./models"):
        os.makedirs("./models")

    # JM Liste der Long Variablen
    Y_str_list = []
    for i in range(args.d_long):
        Y_str = "Y"+str(i+1)
        Y_str_list.append(Y_str)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dag_info_path = f'data/{args.data}_info.pkl' # JM Wird in data_simulation erstellt
    with open(dag_info_path, 'rb') as f:
        dag_info = pickle.load(f)
    
    # JM Hier landen die Results
    d_long = args.d_long
    long_result = {}
    event_ll_result={}
    for i in range(d_long):
        long_result["Y"+str(i+1)] = []
    event_ll_result["visit_ll_mse"] = []
    event_ll_result["surv_ll_mse"] = []
    #surv_result = {}

    pred_window_length = 6 # JM Anzahl der zuk. Zeitpunkte an denen Survival bewertet wird?
    
    
    dt_result = {}
    dt_result["time_err"] = [] # JM Fehler für Visit Zeiten

    seed = args.seed
    seednum = seed
    random.seed(seednum)
    np.random.seed(seednum)
    torch.manual_seed(seednum)

    # load dataset
    dataset_path = f'data/{args.data}_seed_{seed}.pkl' 

    data_all = pd.read_pickle(dataset_path) 
    I = data_all["id"].values[-1]+1

    # JM Logger Print
    logger.info('=' * 50)
    logger.info(f'Starting evaluation for dataset: {args.data}')
    logger.info(f'{args.num_head} heads, {args.num_enc_layer} enc layers,{args.num_dec_layer} dec layers, {args.model_size} model dimension')
    logger.info(f'Data containing total {I} trajectories' )
    logger.info('=' * 50)


    data = data_all[data_all.obstime <= data_all.time]

    # JM Aufteilen der Daten
    random_id = range(I) #np.random.permutation(range(I)) # JM Für reale Daten müsste man nochmal shuffeln 
    train_id = random_id[0:int(0.6*I)]
    vali_id = random_id[int(0.6*I):int(0.8*I)]
    test_id = random_id[int(0.8*I):I]

    train_data = data[data["id"].isin(train_id)]
    vali_data = data[data["id"].isin(vali_id)]
    test_data = data[data["id"].isin(test_id)]

    # Scale data using Min-Max Scaler
    minmax_scaler = MinMaxScaler(feature_range=(-1,1))

    train_data.loc[:,Y_str_list] = minmax_scaler.fit_transform(train_data.loc[:,Y_str_list]) # JM NUR auf die Trainingsdaten gefitted
    vali_data.loc[:,Y_str_list] = minmax_scaler.transform(vali_data.loc[:,Y_str_list])
    test_data.loc[:,Y_str_list] = minmax_scaler.transform(test_data.loc[:,Y_str_list])

    LT = np.quantile(train_data['time'], [0.1] )[0] # JM Landmark time
    # JM Zeitpunkte für die Surv Evaluation zwischen dem 10% und dem 90% Quantil (durch linspace)
    pred_times = np.quantile(train_data['time'].unique(), np.linspace(0.1,0.9,pred_window_length+1))[1:]

    # use all data for accurate censoring distribution JM Es werden alle Daten benutzt um die Zensierungsverteilung zu bekommen
    train_batch = get_tensors(data.copy(),long=Y_str_list)
    train_e,train_t = train_batch["e"].numpy(), train_batch["t"].numpy() # JM train_e ist ob das Event stattgefunden hat und train_t ist dann die Event/Censoring Zeit


    kmf_c = KaplanMeierFitter() # JM Schätzt wie wahrscheinlich es ist bis zu bestimmen Zeitpunkt nicht zensiert zu sein
    # not e to fit for censoring!
    kmf_c.fit(train_t,~train_e)




    model_save_path ='./models/'+args.data+'_'+'seed'+str(seednum)+'_'+args.model+'_'+\
                    "head_"+str(args.num_head)+'_'+"enc_layer_"+str(args.num_enc_layer)+'_'+"dec_layer_"+str(args.num_dec_layer)+'_'+"size_"+str(args.model_size)+\
                        '_'+"visit_weight_"+str(args.inten_weight)+'_'+"surv_weight_"+str(args.surv_weight)+\
                            '_'+"lr_"+str(args.lr)+"Y1miss_"+str(args.Y1_missing)+"Y2miss_"+str(args.Y2_missing)+"Y3miss_"+str(args.Y3_missing)+'.pt'
    # process missing data here
    Y1_nan_inds = np.random.choice(test_data.index,size = int(args.Y1_missing*len(test_data)),replace=False)
    Y2_nan_inds = np.random.choice(test_data.index,size = int(args.Y2_missing*len(test_data)),replace=False)
    Y3_nan_inds = np.random.choice(test_data.index,size = int(args.Y3_missing*len(test_data)),replace=False)


    # JM Create Model (läd das Modell unten aus main.py)
    model = TransformerLSR(d_long=args.d_long,d_base=3,dag_info=dag_info, d_model=args.model_size, nhead=args.num_head,
                num_encoder_layers=args.num_enc_layer,num_decoder_layers=args.num_dec_layer,device=device)
    
    # JM Wenn wir mit Missing machen, dann werden jetzt die NaN werte gesetzt
    if args.model == "LSR_missing":
        test_data["Y1"][Y1_nan_inds] = float('nan') 
        test_data["Y2"][Y2_nan_inds] = float('nan') 
        test_data["Y3"][Y3_nan_inds] = float('nan') 
        
    
    # Only keep subjects with survival time > landmark time
    tmp_data = test_data.loc[test_data["time"]>LT,:]

    # Only keep longitudinal observations <= landmark time
    tmp_data = tmp_data.loc[tmp_data["obstime"]<=LT,:]

    # JM Die ID's die dann übrig bleiben
    surv_id = tmp_data["id"].unique()

    tmp_batch = get_tensors(tmp_data.copy(),long=Y_str_list)
    
    model.to(device=device)

    # JM Läd die gepeicherten Gewichte
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    batch_size = args.batch_size
    
    model.eval() # JM Set the model in Evalution Mode (bspw. Dropout aus)

    # long prediction
    # JM Beinhaltet die Fehler
    num_tokens = 0
    temp_result = {}
    for i in range(d_long):
        temp_result["Y"+str(i+1)+"err"] = 0
        temp_result["Y"+str(i+1)+"tokens"] = 0
    
    temp_result["surv_ll_err"] = 0
    temp_result["surv_ll_tokens"] = 0
    temp_result["visit_ll_err"] = 0
    temp_result["visit_ll_tokens"] = 0

    # JM Schleife für die Brier Scores
    for i in range(pred_window_length):
        temp_result["brier"+"score"+str(i+1)] = 0
    temp_result["ibs"] = 0 # Integrated Brier Score


    for batch in range(0, len(test_id), batch_size):

        indices = test_id[batch:batch+batch_size]
        batch_data = test_data[test_data["id"].isin(indices)]
        batch  = get_tensors(batch_data.copy(),long=Y_str_list,device=device)

        # JM Ist nur schneller (berechnet die Grads nicht)
        with torch.no_grad():
            long_preds = model.predict_next_long_treat(batch)
        
        mask = batch["mask"][:,1:]
        long = batch["long"][:,1:] # JM long. Messwerte (aber noch runterskaliert)
        long_missing = batch["long"][:,1:]
        _batch_size,long_dim,length = long.shape[0],long.shape[-1],long.shape[1]
        nan_mask = torch.isnan(long_missing)
        y_target = torch.clone(long)
        target_mask  = mask.unsqueeze(-1).repeat(1,1,long_dim) # JM Maske auf die richtige Form bringen
        reverseNan_mask =  ~nan_mask
        combined_mask = reverseNan_mask & target_mask # JM Echter visit und kein NaN // Die Maske ist nur noch fürn Fehler dann da


        #inverse transform here
        y_hat = long_preds.cpu().numpy()
        y_target = y_target.cpu().numpy()
        combined_mask = combined_mask.cpu().numpy()
        nan_mask_copy = nan_mask.cpu().numpy()
        y_hat[nan_mask_copy] = 0.0 # JM Dort wo NaN oder kein echter Visit ist wollen wir ja keine Fehler messen (wie auch sinnvoll?)
        y_target[nan_mask_copy] = 0.0
        y_hat = minmax_scaler.inverse_transform(y_hat.reshape(_batch_size*length,long_dim)) # JM Auf die Originalskala zurück transformieren
        y_target = minmax_scaler.inverse_transform(y_target.reshape(_batch_size*length,long_dim))
        y_hat = y_hat.reshape(_batch_size,length,long_dim)
        y_target = y_target.reshape(_batch_size,length,long_dim)
        
        for i in range(d_long): # JM Wir behalten nur gültige Werte
            y_hat_i = y_hat[:,:,i].reshape(-1)[combined_mask[:,:,i].reshape(-1) > 0]
            y_target_i = y_target[:,:,i].reshape(-1)[combined_mask[:,:,i].reshape(-1) > 0]
            temp_result["Y"+str(i+1)+"err"] += np.sum((y_hat_i-y_target_i)**2)
            temp_result["Y"+str(i+1)+"tokens"] += combined_mask[:,:,i].sum().item() # JM Wie viele Werte wurde dafür verwendet

        # now visit event comparison with the ground truth:
        batch  = get_tensors_likelihood(batch_data.copy(),long=Y_str_list,device=device)
        with torch.no_grad():
            _,visit_inten,surv_inten,Lambda,Zeta = model(batch) # JM Berechnet die visit_inten an den Visit Zeitpunkten un die surv_inten am Event
        
        visit_ll_se,surv_ll_se = MSE_likelihood(visit_inten,Lambda,surv_inten,Zeta,batch)
        temp_result["visit_ll_err"] += visit_ll_se # JM Summiert den Fehler auf
        temp_result["surv_ll_err"] += surv_ll_se
        visit_ll_mask = batch["mask"]
        temp_result["visit_ll_tokens"] += visit_ll_mask.sum().item() # JM Anzahl der visits
        # batch size for surv
        temp_result["surv_ll_tokens"] += visit_ll_mask.shape[0] # Anzahl der Patienten im Batch für RMSE

    for i in range(d_long):
        temp_result["Y"+str(i+1)+"err"] /= temp_result["Y"+str(i+1)+"tokens"]
        temp_result["Y"+str(i+1)+"err"] = np.sqrt(temp_result["Y"+str(i+1)+"err"].item()) # JM jetzt RMSE


    temp_result["visit_ll_err"] /= temp_result["visit_ll_tokens"] # JM Wieder mitteln
    temp_result["surv_ll_err"] /= temp_result["surv_ll_tokens"] # 

    temp_result["visit_ll_err"] = np.sqrt(temp_result["visit_ll_err"].item()) # JM RMSE
    temp_result["surv_ll_err"] = np.sqrt(temp_result["surv_ll_err"].item()) 


    # survival analysis
    total_pred = []
    
    for batch in range(0, len(surv_id), batch_size):
        # JM Nur die Patienten die für die Analyse geeignet sind time > LT
        indices = surv_id[batch:batch+batch_size] 
        batch_data = tmp_data[tmp_data["id"].isin(indices)]
        batch = get_tensors(batch_data.copy(),long=Y_str_list,device=device,eval_mode=True)

        base_0 = batch["base"][:,0,:].unsqueeze(1)
        _batch_size = base_0.shape[0]        

        ### Wird nicht benutzt?
        mask_T = torch.ones((_batch_size,1), dtype=torch.bool,device=device)
        time_extender = torch.ones([_batch_size,1],dtype=torch.float32,device=device)
        long_extender = torch.zeros([_batch_size,1,batch["long"].shape[2]],dtype=torch.float32,device=device)
        ### Wird nicht benutzt?
        
        surv_pred = torch.zeros(_batch_size,0,1,device=device)

        start_time = LT

        for pt in pred_times: # JM Berechnet die Integrale immer zwischen den Intervallen
            with torch.no_grad():
                surv_out = model.predict_surv_marginal(batch,end_time=pt,start_time=start_time) # JM noch keine Surv-Wkt nur das Integral h(t) hier
            
            surv_pred = torch.cat((surv_pred, surv_out.unsqueeze(-1)), dim=1)
            start_time = pt

        surv_pred = surv_pred.squeeze().cpu().numpy().reshape(_batch_size,-1)
        surv_pred = surv_pred.cumsum(axis=1) # JM SUmmieren
        surv_pred = np.exp(-surv_pred) # JM Survival WKT jetzt
        total_pred.append(surv_pred) # JM Hier sind jetzt nach und nach aufsummiert die SURV-WKT drin (Integral immer von LT aus und nach und nach mit pt weiter)
    
    total_pred = np.concatenate(total_pred,axis=0)

    # JM Brier Score für jede prediction Zeit
    bs= brier_fast(total_pred, tmp_batch["e"].numpy().reshape(len(surv_id)), tmp_batch["t"].numpy().reshape(len(surv_id)),
                    kmf_c, LT, pred_times)

    # Durschnitt des Bries Scores über die Zeit (Durchschnittlicher Fehler über den gesamten Vorhersagehorizont)
    ibs = get_integrated(bs,pred_times)


    # JM Save everything
    for i in range(len(bs.reshape(-1))):
            temp_result["brier"+"score"+str(i+1)] =bs.reshape(-1)[i]

    temp_result["ibs"]=ibs

   
    eval_result_path = './results/'+args.data+'_'+'seed'+str(seednum)+'_'+args.model+'_'+\
                    "head_"+str(args.num_head)+'_'+"enc_layer_"+str(args.num_enc_layer)+'_'+"dec_layer_"+str(args.num_dec_layer)+'_'+"size_"+str(args.model_size)+\
                        '_'+"visit_weight_"+str(args.inten_weight)+'_'+"surv_weight_"+str(args.surv_weight)+\
                            '_'+"lr_"+str(args.lr)+"Y1miss_"+str(args.Y1_missing)+"Y2miss_"+str(args.Y2_missing)+"Y3miss_"+str(args.Y3_missing)+'.pkl'
    with open(eval_result_path, 'wb') as f:
        pickle.dump(temp_result,f)
            

if __name__ == '__main__':
    main()

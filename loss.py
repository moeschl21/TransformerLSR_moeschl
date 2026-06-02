import torch
import torch.nn as nn
import numpy as np

# handles missing data
# JM Calculates the MSE-Loss fo rlongitudinal Variables (Y_1 bis Y_3)
def long_loss_LSR(long_hat,batch):
    visit_mask = batch["mask"][:,1:]
    long = batch["long"][:,1:]
    long_dim = long.shape[-1] # How many long. Variables
    batch_size,length = long.shape[0],long.shape[1] # batch_size = patients and length = visits
    nan_mask = torch.isnan(long)
    long_target = torch.clone(long)
    long_target[nan_mask] = 0.0
    target_mask = visit_mask.unsqueeze(-1).repeat(1,1,long_dim) # JM Erweitern der Maske auf die longitudinalen Dimensionen
    reverseNan_mask =  ~nan_mask
    combined_mask = reverseNan_mask & target_mask # JM quasi logisches UND und ist die Maske wo kein Padding ist und kein NaN
    long_target = long_target.reshape(-1)[combined_mask.reshape(-1) > 0] # JM we get rid of Padding and NaN
    long_hat = long_hat.reshape(-1)[combined_mask.reshape(-1) > 0]
    long_loss = torch.mean((long_hat-long_target)**2) # JM MSE Loss

    full_loss = torch.sum((long_hat-long_target)**2) # Sum loss for logging
    num_tokens= combined_mask.sum()

    return long_loss,full_loss,num_tokens


# JM (negative) log-likelihood loss for the intensity, wir wollen ja die Parameter so schätzen dass die wkt am höchsten ist, deswegen dann auch negativ
def inten_loss(inten,Lam,batch):
    long_mask = batch["longmask"]
    batch_mask = batch["mask"]
    event_ll = (torch.log(inten)*long_mask[:,1:]).sum(dim=-1) # Jm Berechnetn den Event-Term der Log Likelihood also die Summe
    #use the batch mask -- think about the simple case [event,event,pad (e),e (pad)]
    #the desired mask is indeed [1,1,0] 
    non_event_ll = (Lam * batch_mask).sum(dim=-1) # JM Berechnet die Non-Event-Term der loglikehood (also das Integral welches bestraft wenn das modell einfach nur hohe intensität anzeigt)

    full_loss = event_ll - non_event_ll # JM Log-Likelihood per patient ist entsprechend ein Vektor
    # normalize by total tokens
    ll_loss_full = torch.sum(full_loss)
    ll_loss = ll_loss_full/batch_mask.sum().item()
    nll_loss = -ll_loss # JM negativ weil wir log liklihood maximieren wollen aber Pytorch minimiert
    nll_loss_full = -ll_loss_full
    return nll_loss,nll_loss_full

# JM (negative) log-likelihood loss for the survival (as above)
def surv_loss(inten,Lam,batch):
    event = batch["e"]
    full_mask = batch["fullmask"]
    death_mask = batch["intenmask"] # JM hat nichts mit der Visit Intensität zu tun, markiert den Todeszeitpunkt
    batch_mask = batch["mask"]
    #contribution from possibly the last survival event
    event_ll = (torch.log(inten)*death_mask[:,1:]).sum(dim=-1) # JM log Hazard am Todeszeitpunkt
    #contribution from the intervals 
    #think again about the simple case [visit,visit,e,pad] (full mask: [1,1,1,0])
    #the desired mask is [1,1,0]
    non_event_ll = (Lam * batch_mask).sum(dim=-1) # JM non event part
    full_loss = event_ll - non_event_ll
    # normalize by batch size
    ll_loss_full = torch.sum(full_loss)
    ll_loss = ll_loss_full/batch_mask.sum().item()
    nll_loss = -ll_loss
    nll_loss_full = -ll_loss_full
    return nll_loss,nll_loss_full

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy.stats import expon
import scipy.stats as stats
import pickle
import json
import argparse
import random
import matplotlib.pyplot as plt

# JM Nochmal dann irgendwann mit dem Paper die ganzen Zahlen vergleichen und mal genau rausfinden was hier genau was ist.
#

# JM Reads command-line arguments centrally and then return them after parsing.
def readParser():
    # JM Adding ArgumentParser-object which sepcifies the arguments, what is allowed and default values
    parser = argparse.ArgumentParser(description='synthetic data generation inspired by the DIVAT dataset')
    parser.add_argument('--seed', type=int, default=123, 
                        help='random seed (default: 123)')
    parser.add_argument('--num_traj', type=int, default=1000, # JM Patients
                        help='number of trajectories')
    parser.add_argument('--timeout', type=int, default=1000,
                        help='environment timeout') # JM Maximale visits 
    parser.add_argument('--save', type=str, default='DIVAT_sim',
                    help='data file name')
    parser.add_argument("--plot", action="store_true")   
    return parser.parse_args()


def sigmoid_k(theta_alpha,k):
  return(k/(1+np.exp(theta_alpha)))

# JM "Stepfunction"
def heavi_side(x,val,loc):
    return val * (x > loc) # True = 1

# JM Simulations environment: "Patient" with certain states and parameters
class DIVAT_env:
    # JM Default parameters/Koefficients like sigma/beta etc.
    def __init__(self, sigma2_l=0.1**2,sigma2_d=0.3**2,beta_s1=1,
                        beta_s2=0.9,beta_s3=0,beta_alpha=0,h0=1,omega=1.25,mean_y_init=5,timeout=1000):

        self.time = 0 # JM current simulation time
        self.steps_elapsed = 0
        self.tox = 0

        # JM Random effects from the normal distribution
        self.b_il = stats.multivariate_normal.rvs(mean=np.array([0,0,0]),
                                                  cov=np.array([0.2**2,0.07**2,1*10**(-8)]), size=1)


        self.b_il_tl = stats.multivariate_normal.rvs(mean=np.array([0,0,0]),
                                                  cov=np.array([0.2**2,0.07**2,1*10**(-8)]), size=1)

        # JM Baseline Kovariates
        self.DGF = stats.binom.rvs(n=1, p=0.4, size=1)[0]
        self.ageD = stats.norm.rvs(loc=0, scale=1, size=1)[0]
        self.BMI = stats.norm.rvs(loc=0, scale=1, size=1)[0]
        self.sigma2_l = sigma2_l
        self.sigma2_d = sigma2_d

        # JM Initialization of the longitudinal states
        self.y = stats.norm.rvs(loc=mean_y_init, scale=np.sqrt(self.sigma2_l), size=1)
        
        # JM Saving the start values (useful for reset())
        self.mean_y_init = mean_y_init
        self.mean_tl_init = 2.0
        self.Etl = self.mean_tl_init
        self.tl = stats.norm.rvs(loc=self.mean_tl_init, scale=np.sqrt(self.sigma2_l), size=1) #JM müsste auch long variable sein
        self.Ey = mean_y_init # JM Ground Truth

        # JM Survival/hazard parameters
        self.beta_s=beta_s1
        self.beta_sd=beta_s2
        self.beta_sd_cum=beta_s3
        self.beta_alpha = beta_alpha
        self.shape = omega
        self.h0=h0
        
        # JM I think fixed parameters for the modell
        self.k=2
        self.theta_a=np.array([9.5,-1.5])
        self.beta_l=np.array([3.3,0.1,0.3,0.4, 0.25,1.0, -1*10**(-4),0])
        self.beta_tl = np.array([2.0,0.3,0.1,0.6, 0.2, -1*10**(-4),0])
        self.beta_d=np.array([1,0.2,0.15,0.2,0.15])
        self.timeout = timeout
        self.eta_tox=50
        # JM Censoring rates 
        self.censor_dist = stats.weibull_min(2,scale=8000)
        self.censortime = self.censor_dist.rvs(size=1)[0]
        self.max_visit = 1500 # JM WIe weit wird in die Zukunft nach den nächsten visit geschaut
        
        # gamma intensity parameters
        self.nu = np.exp(2.5)
        self.kappa = np.exp(1.5)+1
        self.mu = -4.8



    # JM Die Funktion rechnet aus wie viel Restwirkung/belastung des Medikaments noch im System sind
    # JM t_upper neuer Zeitpunkt, t_lower alter Zeitpunkt, di Dosis
    def toxicity(self, t_upper, t_lower, di):
        ti = t_upper
        t_r = t_lower
        tox = self.tox*np.exp(-(ti-t_r)/self.eta_tox) # JM Berechnet die Toxizität aus der bisherigen gespeicherten (self.tox und dem rest)
        weight=(1-np.exp(-(ti-t_r)/self.eta_tox))
        tox=tox+di*weight # JM Vergangene tox nimmt ab und der rest nimmt dann zu
        return tox
    
    # JM Berechnet Hazard Rate (h(t)) zum Zeitpunkt t_upper in Abhänigkeit verschiedener Parameter 
    def hazard_fun(self,t_upper,di):
        Zvec_tl = np.array([1,di,self.ageD, self.DGF, self.BMI, self.time,self.time**2]) # JM Fixed Parameter
        Rvec_tl = np.array([1,di,self.time]) # JM Random effects vector
        mean_fixed_tl = np.dot(Zvec_tl, self.beta_tl) 
        mean_rand_tl = np.dot(Rvec_tl, self.b_il_tl) # JM self.beta_tl ist der patienten spezifische random effects Anteil
        Etl = mean_rand_tl+mean_fixed_tl
        Zvec = np.array([1,di,self.ageD, self.DGF, self.BMI,Etl.item(), self.time,self.time**2])
        Rvec = np.array([1,di,self.time])
        mean_fixed = np.dot(Zvec, self.beta_l)
        mean_rand = np.dot(Rvec, self.b_il)
        Ey = mean_rand+mean_fixed
        ti = t_upper
        t_r = self.time
        Etl_diff = (ti-t_r)*(self.beta_tl[-2]+self.b_il_tl[-1])+(ti-t_r)**2*self.beta_tl[-1]
        # expectation taken over tl value
        Ey_diff = (ti-t_r)*(self.beta_l[-2]+self.b_il[-1])+(ti-t_r)**2*self.beta_l[-1] +\
                    Etl_diff*(self.beta_l[5])
        Ey_updated = Ey +Ey_diff
        
        # no alpha
        haz = (self.shape)*np.exp(-(self.beta_s*Ey_updated+self.beta_sd*di+ \
               self.h0))*ti**(self.shape-1)
        
        haz = np.nan_to_num(haz,posinf=0, neginf=-1e32)



        return haz





    # soft reset; keep the same personal data
    # JM Setzt dynamische Zustände der Umgebungen zurück, also Zeit longitudinale Werte, Tox etc.
    def reset(self):
        self.y = stats.norm.rvs(loc=self.mean_y_init, scale=np.sqrt(self.sigma2_l), size=1) # JM Neue Anfangswerte
        self.Ey = self.mean_y_init

        self.Etl = self.mean_tl_init # JM gleiche wie oben
        self.tl = stats.norm.rvs(loc=self.mean_tl_init, scale=np.sqrt(self.sigma2_l), size=1)

        self.time = 0
        self.steps_elapsed=0
        self.tox = 0
        obs = self.get_obs()
        return obs # JM dort sind dann einfach nochmal die Daten auf und gibt sie weiter
    
    # hard reset; redraw the random effect and personal data, or load from argument (JM je Patient)
    # JM die Methode wird benutzt um ein Objekt immer wieder zu resetten und dann so das Datenset aufzubauen, die Daten werden dann nach und nach seperat gespeichert (siehe simulate_traj())
    def hard_reset(self,info_i=None):
        if info_i:
            self.b_il = info_i['b_il']
            self.BMI = info_i['BMI']
            self.ageD = info_i['ageD']
            self.DGF = info_i['DGF']
            self.b_il_tl = info_i['b_il_tl']
        else: # JM Es wird alles neu gezogen
            self.b_il = stats.multivariate_normal.rvs(mean=np.array([0,0,0]),
                                                  cov=np.array([0.2**2,0.07**2,1*10**(-8)]), size=1)
            self.b_il_tl = stats.multivariate_normal.rvs(mean=np.array([0,0,0]),
                                                  cov=np.array([0.2**2,0.07**2,1*10**(-8)]), size=1)
            self.DGF = stats.binom.rvs(n=1, p=0.4, size=1)[0]
            self.ageD = stats.norm.rvs(loc=0, scale=1, size=1)[0]
            self.BMI = stats.norm.rvs(loc=0, scale=1, size=1)[0]
        obs = self.reset() # JM noch die Anfangsmessungen etc auch zurückzusetzen
        self.censortime = self.censor_dist.rvs(size=1)[0]
        return obs

    # JM Get all info about the patient (goes into PatientInfo.pkl)
    def get_data_i(self):
        data_info = {}
        data_info['BMI'] = self.BMI
        data_info['ageD'] = self.ageD
        data_info['DGF'] = self.DGF
        data_info['b_il'] = self.b_il
        data_info['b_il_tl'] = self.b_il_tl
        return data_info

    def get_obs(self):
        obs = [self.tl.item(),self.y.item()]
        return obs
    
    # "thinning" algorithm for survival  (JM check supplements for thinning algo)
    # JM accepted means patient died 
    def sample_event(self,di,delta_t):
        t_initial=self.time
        t_lower = self.time
        t_upper = self.time+delta_t
        #candidates from a homogeneous Poisson process
        t_candidate = t_lower
        
        # search for lambda upper bound in [t_lower,t_upper]
        t_range = np.linspace(t_lower+1e-8,t_upper,num=200)
        lambda_range=self.hazard_fun(t_range,di) 
        
        lambda_bar = np.max(lambda_range)
        
        accepted = False
        t_accept = None
        if lambda_bar ==0:
            return accepted, t_accept
        # by here, lambda_bar >0 
        while t_lower<t_upper:
            #t_bar ~ exp(lambda_bar)
            t_bar = expon.rvs(scale=1/lambda_bar,size=1)[0]
            t_candidate =t_lower+t_bar
            
            lambda_candidate = self.hazard_fun(t_candidate,di)
            u_s = stats.uniform.rvs(size=1)
            
            if u_s <= (lambda_candidate/lambda_bar) and t_candidate<t_upper:
                t_accept = t_candidate
                accepted = True
                break
            t_lower = t_candidate
        return accepted, t_accept

   
    # JM Calculate the integrals of intensity and hazard functions
    def cumulative_intensity(self, t_upper,di):
        I = integrate.quad(self.intensity_fun, self.time, t_upper, args=(di) )[0] #JM bei 0 liegt nur der Integralwert, bei [1] der Fehler
        cumu_prob = 1-np.exp(-I)
        #return(cumu_prob)
        return I
    def cumulative_prob(self, t_upper, t_lower, di):
        I = integrate.quad(self.hazard_fun, t_lower, t_upper, args=(di) )[0]
        cumu_prob = 1-np.exp(-I)
        #return(cumu_prob)
        return I



    # JM Berechnet Intensity Rate (lambda(t)) zum Zeitpunkt t_upper in Abhänigkeit verschiedener Parameter 
    def intensity_fun(self,t_upper,di):
        Zvec_tl = np.array([1,di,self.ageD, self.DGF, self.BMI, self.time,self.time**2])
        Rvec_tl = np.array([1,di,self.time])
        mean_fixed_tl = np.dot(Zvec_tl, self.beta_tl)
        mean_rand_tl = np.dot(Rvec_tl, self.b_il_tl)
        Etl = mean_rand_tl+mean_fixed_tl
        Zvec = np.array([1,di,self.ageD, self.DGF, self.BMI,Etl.item(), self.time,self.time**2])
        Rvec = np.array([1,di,self.time])
        mean_fixed = np.dot(Zvec, self.beta_l)
        mean_rand = np.dot(Rvec, self.b_il)
        Ey = mean_rand+mean_fixed
        ti = t_upper
        t_r = self.time
        Etl_diff = (ti-t_r)*(self.beta_tl[-2]+self.b_il_tl[-1])+(ti-t_r)**2*self.beta_tl[-1]
        # expectation taken over tl value
        Ey_diff = (ti-t_r)*(self.beta_l[-2]+self.b_il[-1])+(ti-t_r)**2*self.beta_l[-1] +\
                    Etl_diff*(self.beta_l[5])
        Ey_updated = Ey +Ey_diff
        

        lam_0 = 3
        nu_1 = 1

        # no di dependence for now
        nu_2 = 0
        nu_0 = 1.5        

        lam = lam_0*np.exp(-(nu_1*Ey_updated+nu_2*di\
               +nu_0)) \
        *ti**(self.shape-1) # JM anscheinend wie weibull 
        return lam


    # "thinning" algorithm for recurrent visits (zeitpunkte) (JM check supplements for thinning algo)
    def sample_visit(self,di):
        t_lower = self.time
        t_upper = self.time+self.max_visit
        #candidates from a homogeneous Poisson process
        t_candidate = t_lower
        
        # search for lambda upper bound in [t_lower,t_upper]
        t_range = np.linspace(t_lower+1e-8,t_upper,num=200)

        lambda_range=self.intensity_fun(t_range,di) 
        
        
        lambda_bar = np.max(lambda_range)
        
        accepted = False
        t_accept = None
        if lambda_bar ==0:
            return accepted, t_accept
        # by here, lambda_bar >0 
        # keep track of t_bar for debug purposes
        t_bar_vec = []
        u_vec = []
        lam_cands_vec = []
        while t_lower<t_upper:
            #t_bar ~ exp(lambda_bar)
            t_bar = expon.rvs(scale=1/lambda_bar,size=1)[0]
            t_bar_vec.append(t_bar)
            
            t_candidate =t_lower+t_bar
            
            lambda_candidate = self.intensity_fun(t_candidate,di)
            lam_cands_vec.append(lambda_candidate)
            u_s = stats.uniform.rvs(size=1)
            u_vec.append(u_s)
            if u_s <= (lambda_candidate/lambda_bar) and t_candidate<t_upper:
                t_accept = t_candidate
                accepted = True
                break
            t_lower = t_candidate

        return accepted, t_accept


    # measurement dependent treatment assignment and gives back dosage and next visit time and the intensity at the next visit
    def sample_treatment(self):
        #update alpha first
        #self.alpha = sigmoid_k(np.dot(self.theta_a,np.array([1,self.y.item()])),self.k)
        cov_d = np.array([1,self.y.item(),self.ageD,self.DGF,self.BMI])
        dosage_mean =np.dot(cov_d,self.beta_d) 
        dosage = stats.norm.rvs(loc=dosage_mean, scale=np.sqrt(self.sigma2_d), size=1).item()
        accepted,time = self.sample_visit(dosage)
        if not accepted:
            time = self.max_visit # JM Kommt aus dem thinning, wenn nicht berechnet, dann maximale Wartezeit
        else:
            time = time - self.time
        inten_next = self.intensity_fun(self.time+time,dosage).item()
        return dosage,time,inten_next



    # have Ey depend on the actual outcome of tl for now
    # JM updates longitudinal variables for time t_ij
    def update_long(self,di,t_ij,update_obs=False):
        
        Zvec_tl = np.array([1,di,self.ageD, self.DGF, self.BMI, t_ij,t_ij**2]) # JM Das müsste Tacrolimus sein
        Rvec_tl = np.array([1,di,t_ij])
        mean_fixed_tl = np.dot(Zvec_tl, self.beta_tl)
        mean_rand_tl = np.dot(Rvec_tl, self.b_il_tl)
        self.Etl = mean_rand_tl+mean_fixed_tl

        Zvec = np.array([1,di,self.ageD, self.DGF, self.BMI,self.tl.item(), t_ij,t_ij**2]) # JM Das müsste Kreatinin Y sein
        Rvec = np.array([1,di,t_ij])
        mean_fixed = np.dot(Zvec, self.beta_l)
        mean_rand = np.dot(Rvec, self.b_il)
        self.Ey = mean_rand+mean_fixed
        #update Zvec again!
        if update_obs:
            self.tl = stats.norm.rvs(loc=self.Etl, scale=np.sqrt(self.sigma2_l), size=1)
            Zvec = np.array([1,di,self.ageD, self.DGF, self.BMI,self.tl.item(), t_ij,t_ij**2])
            mean_fixed = np.dot(Zvec, self.beta_l)
            self.Ey = mean_rand+mean_fixed
            self.y = stats.norm.rvs(loc=self.Ey, scale=np.sqrt(self.sigma2_l), size=1)


    # JM Main function for the simulation
    def step(self, action):
        di = action[0]
        delta_t = action[1]

        # update Ey and Etl since di changed
        self.update_long(di,self.time,update_obs=False)

        self.steps_elapsed+=1

        t_ij = self.time+delta_t
        death = False
        #self.alpha = sigmoid_k(np.dot(self.theta_a,np.array([1,self.y.item()])),self.k)
        death, T_max = self.sample_event(di,delta_t)    # JM Samples if the patient dies
        #survive until t_ij
        if not death:
            if t_ij >= self.censortime: # JM Integrale werden nur bis zum Zensurzeitpunkt dann berechnet
                print("censored")
                Zeta = self.cumulative_prob(t_upper=self.censortime,t_lower=self.time,di=di)
                Lambda = self.cumulative_intensity(t_upper=self.censortime,di=di)
                done = True
                reward = self.censortime - self.time
                self.update_long(di,t_ij,update_obs=True)
                self.tox =  self.toxicity(t_ij, self.time, di)
                self.time = self.censortime
                
            else: # JM nicht Tod, dann geht es normal weiter
                Zeta = self.cumulative_prob(t_upper=t_ij,t_lower=self.time,di=di)
                Lambda = self.cumulative_intensity(t_upper=t_ij,di=di)
                done = False
                reward = delta_t
                self.update_long(di,t_ij,update_obs=True)
                self.tox =  self.toxicity(t_ij, self.time, di)
                self.time = t_ij
        else: # JM Integrale nur bis zum Todeszeitpunkt
            Zeta = self.cumulative_prob(t_upper=T_max,t_lower=self.time,di=di)
            Lambda = self.cumulative_intensity(t_upper=T_max,di=di)
            done = True
            self.update_long(di,T_max,update_obs=True)
            reward = T_max - self.time
            self.time = T_max
     
        obs = self.get_obs()
        # do a hard conversion to make sure output reward is scalar
        reward = np.array(reward)
        if self.steps_elapsed == self.timeout:
            done = True
        return obs, reward.item(), done, death, Lambda, Zeta




# Y1 indep (tl) Y2 dep (creat)
# JM creates patient history
def simulate_traj(env):

    _ = env.hard_reset()
    data_info = env.get_data_i()
    death,done = False,False
    Y1,Y2,A = [],[],[] # JM Tacrolimus, Kreatinin und Dosis
    obstime,true_inten = [],[] # JM No times set yet
    true_surv,Lam_vec,Zeta_vec = [],[],[]
    tox = []
    num_visit = 0
    Lam_sum,Zeta_sum = 0,0
    while (not done):
        num_visit += 1
        dosage,dt,inten = env.sample_treatment() # dt ist die Wartezeit bis zum nächsten Besuch, also delta_t
        action = [dosage,dt]
        state = env.get_obs()
        # record data pre step
        true_inten.append(inten)
        Y1.append(state[0])
        Y2.append(state[1])
        A.append(dosage)
        obstime.append(env.time)
        tox.append(env.tox)
        # execute the action
        next_state, reward, done, death, Lambda, Zeta= env.step(action)

        Lam_sum+=Lambda
        Zeta_sum +=Zeta
        Lam_vec.append(Lambda)
        Zeta_vec.append(Zeta)
        haz = env.hazard_fun(env.time,dosage).item()
        true_surv.append(haz)



    true_time = env.time
    if death:
        last_haz = np.log(env.hazard_fun(env.time,dosage)).item()
          
    else:
        last_haz = 0
    # JM ll steht für log likelihood und bspw. surv_non_ll für den nicht Event teil der Survival log likelihood
    surv_ll = last_haz
    surv_non_ll = Zeta_sum

    event_ll = np.sum(np.log(true_inten).reshape(-1)[:-1]).item() # JM es wird die Intensität für den "nächsten besuch also + 1 berechnet, dieser ist natürlich aber kein Echter besuch wenn Patient stirbt oder den letzten Besuch nicht machen kann
    event_non_ll = Lam_sum
    
    # JM Saving everything
    out_dict = {}
    out_dict["Y1"],out_dict["Y2"],out_dict["A"] = Y1,Y2,A
    out_dict["time"],out_dict["obstime"],out_dict["event"] = true_time,obstime,death
    out_dict["X1"],out_dict["X2"],out_dict["X3"],out_dict["num_visit"] = data_info["ageD"],data_info["DGF"],data_info["BMI"],num_visit
    out_dict["true_inten"] = true_inten
    out_dict["true_surv"] = true_surv
    out_dict["Lam"], out_dict["Zeta"] = Lam_vec, Zeta_vec
    #out_dict["last_obs"] = last_obs
    out_dict["last_treat"] = A[-1]
    out_dict["surv_ll"],out_dict["surv_non_ll"],out_dict["event_ll"],out_dict["event_non_ll"] = surv_ll,surv_non_ll, event_ll, event_non_ll
    out_dict["info"] = env.get_data_i()
    out_dict["tox"] = tox

    return out_dict






# JM
def main(args=None):
    if args is None:
        args = readParser()
    seednum = args.seed
    random.seed(seednum)
    np.random.seed(seednum)


    d_long = 3 # JM Count of longitudinal variables

    save_path = 'data/'+args.save +'_'+str(args.num_traj)+'_visit_'+str(args.timeout)+'_long_'+str(d_long)+'_seed_'+str(args.seed)+'.pkl'
    patient_info_path = 'data/'+args.save +'_'+str(args.num_traj)+'_visit_'+str(args.timeout)+'_long_'+str(d_long)+'_seed_'+str(args.seed)+'_patientInfo.pkl'
    I,J = args.num_traj, args.timeout # JM Number of patients and maximum time

    # JM Initialize all lists
    Y1,Y2,A,X1,X2,X3 = [],[],[],[],[],[] # JM X's is Baseline
    time,obstime,event,true_prob,num_visit,true_inten = [],[],[],[],[],[]
    true_surv,Lam,Zeta = [],[],[]
    last_obs,last_treat = [],[]
    tox = []
    surv_ll,surv_non_ll,event_ll,event_non_ll = [],[],[],[]
    ID,visit = [],[]
    env = DIVAT_env(timeout=J)
    # JM helping variables for statistics
    data_info = []
    death_count = 0
    death_time = []
    mean_visit = []
    event_time = []
    cens_time = []
    dt_vec  = []
    patient_info = []
    length_checker = False
    # JM For every patient we are doing a trajectory, alles wird in einem Array, Besuchsweise reingepackt
    for i in range(I):
        while not length_checker:
            result = simulate_traj(env)
            if result["num_visit"]>1:
                length_checker = True  
        ID.extend(np.repeat(i,result["num_visit"]))
        visit.extend(np.arange(0,result["num_visit"]))
        Y1.extend(result["Y1"]),Y2.extend(result["Y2"]),A.extend(result["A"])
        X1.extend(np.repeat(result["X1"],result["num_visit"])),X2.extend(np.repeat(result["X2"],result["num_visit"])),X3.extend(np.repeat(result["X3"],result["num_visit"]))
        time.extend(np.repeat(result["time"],result["num_visit"])),obstime.extend(result["obstime"]),event.extend(np.repeat(result["event"],result["num_visit"]))
        num_visit.extend(np.repeat(result["num_visit"],result["num_visit"]))
        true_inten.extend(result["true_inten"])
        true_surv.extend(result["true_surv"])
        Lam.extend(result["Lam"])
        Zeta.extend(result["Zeta"])
        #last_obs.append(result["last_obs"])
        last_treat.extend(np.repeat(result["last_treat"],result["num_visit"]))
        surv_ll.extend(np.repeat(result["surv_ll"],result["num_visit"]))
        surv_non_ll.extend(np.repeat(result["surv_non_ll"],result["num_visit"]))
        event_ll.extend(np.repeat(result["event_ll"],result["num_visit"]))
        event_non_ll.extend(np.repeat(result["event_non_ll"],result["num_visit"]))
        data_info.append(result["info"])
        tox.extend(result["tox"])
        diff_time = np.diff(result["obstime"])
        dt_vec.append(diff_time.reshape(-1))

        patient_info.append(result["info"])

        if result["event"]:
            death_count+=1
            death_time.append(result["time"])
        else:
            cens_time.append(result["time"])
        event_time.append(result["time"])
        mean_visit.append(result["num_visit"])

        length_checker = False


    death_time = np.array(death_time).reshape(-1)
    mean_visit = np.array(mean_visit).reshape(-1)
    event_time = np.array(event_time).reshape(-1)
    cens_time = np.array(cens_time).reshape(-1)
    
    # JM wenn angegeben, dann kann man hier ein Histogramm plotten lassen
    if args.plot:
        plt.hist(death_time)
        plt.savefig("plots/death_time_hist.png")
        plt.close()
        plt.hist(event_time)
        plt.savefig("plots/event_time_hist.png")
        plt.close()

        plt.hist(mean_visit)
        plt.savefig("plots/visit_hist.png")
        plt.close()

        plt.hist(cens_time)
        plt.savefig("plots/cens_hist.png")
        plt.close()

    # JM Formatierung in die richtigen Datenformate
    #Y1,Y2,Y3,Y4 = np.array(Y1,dtype=np.float64),np.array(Y2,dtype=np.float64),np.array(Y3,dtype=np.float64),np.array(Y4,dtype=np.float64)
    Y1,Y2,A = np.array(Y1,dtype=np.float64),np.array(Y2,dtype=np.float64),np.array(A,dtype=np.float64)
    X1,X2,X3 = np.array(X1).astype(np.float64), np.array(X2).astype(np.float64),np.array(X3).astype(np.float64)
    num_visit = np.array(num_visit).astype(np.int32) 
    time,event = np.array(time).astype(np.float64), np.array(event).astype(bool)
    #last_obs = np.repeat(last_obs,repeats=J).astype(np.float64)
    last_treat = np.array(last_treat).astype(np.float64)
    obstime = np.array(obstime).astype(np.float64)
    true_inten = np.array(true_inten).astype(np.float64)
    true_surv = np.array(true_surv).astype(np.float64)
    Zeta = np.array(Zeta).astype(np.float64)
    Lam = np.array(Lam).astype(np.float64)
    ID,visit = np.array(ID).astype(np.int32),np.array(visit).astype(np.int32)
    surv_ll,surv_non_ll,event_ll,event_non_ll = np.array(surv_ll).astype(np.float64),np.array(surv_non_ll).astype(np.float64),\
                                np.array(event_ll).astype(np.float64),np.array(event_non_ll).astype(np.float64)
    tox = np.array(tox,dtype=np.float64)

    # JM Prints Statistics
    print(f"death rate: {death_count/args.num_traj}")
    
    death_time = np.sum(death_time)/death_count
    mean_visit = np.sum(mean_visit)/args.num_traj
    
    
    print(f"mean death time: {death_time}")
    print(f"mean visit: {mean_visit}")

    dt_vec = np.concatenate(dt_vec,axis=0)
    diff_mean = np.mean(dt_vec)
    diff_std = np.std(dt_vec)

    print(f"mean time difference :{diff_mean.item():.2f}")
    print(f"Data time difference std:{diff_std.item():.2f}")


    data = pd.DataFrame({"id":ID, "visit":visit, "obstime":obstime, 
                        "time":time, "last_treat":last_treat,
                        "event":event,"num_visit":num_visit,
                        "Y1":Y1,"Y2":Y2,"Y3":A,
                        "X1":X1,
                        "X2":X2,"X3":X3,"true_inten":true_inten,
                        "true_surv":true_surv,"Lam":Lam,"Zeta":Zeta,
                        "surv_ll":surv_ll,"surv_non_ll":surv_non_ll,"event_ll":event_ll,"event_non_ll":event_non_ll,
                        "tox":tox})
    pd.to_pickle(data, save_path)  
    
    
    dag_mat = np.array([[0,0,0],[1,0,0],[1,1,0]])
    dag_mat = dag_mat[:d_long,:d_long]

    # JM DAG = Directed Acyclic Graph Gibt die Abhänigkeiten der Variablen an also Y1 beinflusst Y2 und Y3 und Y2 beinfluss Y3 
    dag_mat = dag_mat.transpose()
    dag_info={}
    dag_info["dag"]= dag_mat
    
    dag_order = []
    
    for i in range(d_long):
        dag_order.append(i)
    
    dag_info["order"] = dag_order

    # JM Speichert DAG Info und Patientinfos nochmal
    info_path = 'data/'+args.save +'_'+str(args.num_traj)+'_visit_'+str(args.timeout)+'_long_'+str(d_long)+'_info.pkl'
    with open(info_path, 'wb') as f:
        pickle.dump(dag_info,f)
    
    with open(patient_info_path, 'wb') as f:
        pickle.dump(patient_info,f)
    
    print("complete")




if __name__ == '__main__':
    main()

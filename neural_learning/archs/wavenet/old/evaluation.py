from dask import dataframe as dd, array as da
import pandas as pd, numpy as np, warnings
from .files import import_data, save_data
from .wavenet import matrix_G, normalize
from .plotting import plotting
from numba import njit
from tqdm import tqdm

@njit(cache=True, fastmath=True, nogil=True)
def generate_oscillatory(dic, n_Iapps, I_mean, white_noise):

    Iapps = np.zeros(n_Iapps)
    h = dic['h']
    tau = dic['tau']
    mu = dic['mu_est']
    w = dic['w']
    sigma = dic['sigma']
    x0 = I_mean; I0 = I_mean
    Iapps[0] = I0
    
    for t in range(n_Iapps-1):
        I1 = I0 + h * ((x0 + mu*np.cos(w*t) - I0)/tau + sigma*white_noise[t])
        Iapps[t+1] = I1
        I0 = I1
    return Iapps
    
def evaluation(param, euler_dict, model, var, title, **IC_eval):
    # Defines a desidered Iapp input signal
    if param['generateIapp_eval_oscillatory']:
        n_Iapps = param['n_Iapp_eval']*param['integrations_eval']
        white_noise = np.random.RandomState(80085).normal(0,1, n_Iapps-1)
        I_mean = (param['I_max_eval']+param['I_min_eval'])/2
        Iapps = generate_oscillatory(euler_dict, n_Iapps, I_mean, white_noise)
    elif param['generateIapp_eval_stepwise']:
        Iapps = np.random.RandomState(80085).uniform(param['I_min_eval'], param['I_max_eval'], param['n_Iapp_eval'])
    else:
        Iapps = import_data(param)
        if param['shuffle']: np.random.RandomState(80085).shuffle(Iapps)
        Iapps = Iapps[:param['n_Iapp_eval']]
    
    # Generates the target
    euler_dict['integrations'] = param['integrations_eval']
    _, target = model(euler_dict, Iapps, euler_dict['noise'], **IC_eval)
    Iapps = np.array([I for I in Iapps for n_times in range(param['integrations_eval'])])
    save_data('input_eval.parquet', da.from_array(Iapps))
    tuples = [(CI, target[i]) for i,CI in enumerate(IC_eval.values())]
    
    # Prediction and plotting
    prediction(param, var, Iapps, tuples)    
    plotting(param, title, var, Iapps, target, euler_dict['h'])
    
def prediction(param, var, Iapps, tuples):
    d_var={}
    for i,elem in enumerate(tuples):
        d_var['weights_'+str(i)] = np.array(pd.read_parquet(param['results_folder']+'/weights_' + var[i] + '.parquet'))
        d_var['t_'+str(i)] = elem[1]
        d_var['i_'+str(i)] = np.array([elem[0]])
        d_var['predicted_'+str(i)] = np.zeros_like(elem[1])

    for j,I in enumerate(tqdm(Iapps, desc='Prediction', unit=' integrations', leave=True)):
        normalized = normalize(param, (d_var['i_'+str(i)] for i in range(len(tuples))), np.array([I]))
        G = matrix_G(param, normalized)
        for i in range(len(tuples)):
            d_var['i_'+str(i)] = (G.dot(d_var['weights_'+str(i)]))[0] #n+1
            d_var['predicted_'+str(i)][j] = d_var['i_'+str(i)]
          
    warnings.simplefilter('ignore') #ignore numeric errors, later seen in the metrics
    for i in range(len(tuples)):
        try:
            x1 = d_var['t_'+str(i)][param['rmse_eval']:]
            x2 = d_var['predicted_'+str(i)][param['rmse_eval']:]
            
            rmse = np.sum((x1-x2)**2)/np.sum((np.mean(x1)-x1)**2)*100
            print('->', var[i],'1-r²:', rmse,'%')
            cos = np.inner(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
            print('->', var[i],'Cosine similarity:', cos)
            
        except RuntimeWarning: pass
    
        save_data(param['results_folder']+'/predicted_' + var[i] + '.parquet', da.from_array(d_var['predicted_'+str(i)]))
        save_data(param['results_folder']+'/target_' + var[i] + '.parquet', da.from_array(d_var['t_'+str(i)]))

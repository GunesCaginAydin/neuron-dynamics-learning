from numba import njit, types
from numba.typed import Dict
import numpy as np, sys

from Wavenet.evaluation import evaluation
from configuration import config
from Wavenet.training import training

@njit(cache=True, fastmath=True, nogil=True)
def MLecar(dic, Iapps, noise, w0, v0):
    n_integrations = int(dic['integrations'])
    ## Input variables
    w_train = np.zeros(len(Iapps)*n_integrations)
    v_train = np.zeros(len(Iapps)*n_integrations)
    ## Targets
    target_w = np.zeros(len(Iapps)*n_integrations)
    target_v = np.zeros(len(Iapps)*n_integrations)
    ## Parameters
    h = dic['h']
    Cm = dic['Cm']
    phi = dic['phi']
    EL = dic['EL']
    EK = dic['EK']
    ECa = dic['ECa']
    V1 = dic['V1']
    V2 = dic['V2']
    V3 = dic['V3']
    V4 = dic['V4']
    gL = dic['gL']
    gK = dic['gK']
    gCa = dic['gCa']

    np.random.seed(1)
    noise_1 = np.random.uniform(0, noise, len(Iapps)*n_integrations)
    noise_2 = np.random.uniform(0, noise, len(Iapps)*n_integrations)
    gK = gK*(1+noise_1)
    gL = gL*(1+noise_2)
    
    ## Euler
    i=0
    for n,I in enumerate(Iapps):
        wn = np.zeros(n_integrations+1)
        vn = np.zeros(n_integrations+1)
        wn[0] = w0
        vn[0] = v0
        for k in range(n_integrations):            
            minf = (1+np.tanh((v0-V1)/V2))/2
            winf = (1+np.tanh((v0-V3)/V4))/2
            tauinf = 1/(np.cosh((v0-V3)/(2*V4)))
            # Leakage current
            iL = gL[n_integrations*n:n_integrations*(n+1)][k]*(v0-EL)
            #iL = gL*(v0-EL)
            # Calcium current
            iCa = gCa*minf*(v0-ECa)
            # Potassium current
            iK = gK[n_integrations*n:n_integrations*(n+1)][k]*w0*(v0-EK)
            #iK = gK*w0*(v0-EK)
            # ODE's system
            v1 = v0 + h * (1/Cm)*(-iL-iK-iCa+I)
            w1 = w0 + h * phi*(winf-w0)/tauinf
            wn[k+1] = w1
            vn[k+1] = v1
            v0=v1
            w0=w1
        for j in range(n_integrations):
            target_w[i+j] = wn[1:][j]
            target_v[i+j] = vn[1:][j]
            w_train[i+j] = wn[:-1][j]
            v_train[i+j] = vn[:-1][j]
        i+=n_integrations
    return (w_train, v_train), (target_w, target_v)

if __name__ == "__main__":
    
    ## Config
    param, jit_dict = config(sys.argv[0].split('.')[0])
    var = ['w', 'v']
    title_plots = 'Morris-Lecar'
    
    if not param['only_evaluation']:
        training(param, jit_dict, MLecar, var, w0=param['w0_train'],
                                               v0=param['v0_train'])
    evaluation(param, jit_dict, MLecar, var, title_plots, w0=param['w0_eval'],
                                                          v0=param['v0_eval'])

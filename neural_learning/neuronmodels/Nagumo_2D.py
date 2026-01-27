from numba import njit, types
from numba.typed import Dict
import numpy as np, sys

from Wavenet.evaluation import evaluation
from configuration import config
from Wavenet.training import training

@njit(cache=True, fastmath=True, nogil=True)
def Nagumo_2D(dic, Iapps, noise, w0, v0):
    n_integrations = int(dic['integrations'])
    ## Input variables
    w_train = np.zeros(len(Iapps)*n_integrations)
    v_train = np.zeros(len(Iapps)*n_integrations)
    ## Targets
    target_w = np.zeros(len(Iapps)*n_integrations)
    target_v = np.zeros(len(Iapps)*n_integrations)
    ## Parameters
    h = dic['h']
    a = dic['a']
    eps = dic['eps']
    gamma = dic['gamma']

    np.random.seed(1)
    noise_1 = np.random.uniform(0, noise, len(Iapps)*n_integrations)
    noise_2 = np.random.uniform(0, noise, len(Iapps)*n_integrations)   
    na = a*(1+noise_1)
    ng = gamma*(1+noise_2)
    
    ## Euler
    i=0
    for n,I in enumerate(Iapps):
        wn = np.zeros(n_integrations+1)
        vn = np.zeros(n_integrations+1)
        wn[0] = w0
        vn[0] = v0
        for k in range(n_integrations):
            a = na[n_integrations*n:n_integrations*(n+1)][k]     # na --> a
            gamma = ng[n_integrations*n:n_integrations*(n+1)][k] # ng --> gamma
            v1 = v0 + h * (-v0*(v0-1)*(v0-a)+w0+I)               #dv/dt = -f(v)+w+Ibase
            w1 = w0 + h * eps*(-v0-gamma*w0)                     #dw/dt = eps*(-v-gamma*w)
            wn[k+1] = w1                                         #f(v) = v*(v-1)*(v-a)
            vn[k+1] = v1
            v0=v1
            w0=w1
        for j in range(n_integrations):
            target_w[i+j] = wn[1:][j]#n+1
            target_v[i+j] = vn[1:][j]
            w_train[i+j] = wn[:-1][j]#n
            v_train[i+j] = vn[:-1][j]
        i+=n_integrations
    return (w_train, v_train), (target_w, target_v)

if __name__ == "__main__":
    
    ## Config
    param, jit_dict = config(sys.argv[0].split('.')[0])
    var = ['w', 'v']
    title_plots = 'FitzHugh-Nagumo'
    
    if not param['only_evaluation']:
        training(param, jit_dict, Nagumo_2D, var, w0=param['w0_train'],
                                                  v0=param['v0_train'])
    evaluation(param, jit_dict, Nagumo_2D, var, title_plots, w0=param['w0_eval'],
                                                             v0=param['v0_eval'])

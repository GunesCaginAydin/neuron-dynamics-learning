from numba import njit, types
from numba.typed import Dict
import numpy as np, sys

from Wavenet.evaluation import evaluation
from configuration import config
from Wavenet.training import training

@njit(cache=True, fastmath=True, nogil=True)
def Nagumo_3D(dic, Iapps, noise, w0, v0, y0):
    n_integrations = int(dic['integrations'])
    ## Input variables
    v_train = np.zeros(len(Iapps)*n_integrations)
    w_train = np.zeros(len(Iapps)*n_integrations)
    y_train = np.zeros(len(Iapps)*n_integrations)
    ## Targets
    target_v = np.zeros(len(Iapps)*n_integrations)
    target_w = np.zeros(len(Iapps)*n_integrations)
    target_y = np.zeros(len(Iapps)*n_integrations)
    ## Parameters
    h = dic['h']
    a = dic['a']
    alpha = dic['alpha']
    eps = dic['eps']
    gamma = dic['gamma']
    c = dic['c']
    d = dic['d']
    mu = dic['mu']

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
        yn = np.zeros(n_integrations+1)
        wn[0] = w0
        vn[0] = v0
        yn[0] = y0
        for k in range(n_integrations):
            a = na[n_integrations*n:n_integrations*(n+1)][k]     # na --> a
            gamma = ng[n_integrations*n:n_integrations*(n+1)][k] # ng --> gamma
            v1 = v0 + h * (-v0*(v0-1)*(v0-a)+w0-alpha*y0+I)      #dv/dt = -f(v)+w-alpha*y+Ibase
            w1 = w0 + h * eps*(-v0-gamma*w0)                     #dw/dt = eps*(-v-gamma*w)
            y1 = y0 + h * mu*(c-v0-d*y0)                         #dy/dt = mu*(c-v-d*y)
            wn[k+1] = w1                                         #f(v) = v*(v-1)*(v-a)
            vn[k+1] = v1
            yn[k+1] = y1
            v0=v1
            w0=w1
            y0=y1
        for j in range(n_integrations):
            target_w[i+j] = wn[1:][j]
            target_v[i+j] = vn[1:][j]
            target_y[i+j] = yn[1:][j]
            w_train[i+j] = wn[:-1][j]
            v_train[i+j] = vn[:-1][j]
            y_train[i+j] = yn[:-1][j]
        i+=n_integrations
    return (w_train, v_train, y_train), (target_w, target_v, target_y)

if __name__ == "__main__":
    
    ## Config
    param, jit_dict = config(sys.argv[0].split('.')[0])
    var = ['w', 'v', 'y']
    title_plots = 'FitzHugh-Nagumo 3D'
    
    if not param['only_evaluation']:
        training(param, jit_dict, Nagumo_3D, var, w0=param['w0_train'],
                                                  v0=param['v0_train'],
                                                  y0=param['y0_train'])
    evaluation(param, jit_dict, Nagumo_3D, var, title_plots, w0=param['w0_eval'],
                                                             v0=param['v0_eval'],
                                                             y0=param['y0_eval'])

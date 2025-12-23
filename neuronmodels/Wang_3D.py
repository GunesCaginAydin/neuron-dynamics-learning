from numba import njit, types
from numba.typed import Dict
import numpy as np, sys

from Wavenet.evaluation import evaluation
from configuration import config
from Wavenet.training import training

@njit(cache=True, fastmath=True, nogil=True)
def Wang_3D(dic, Iapps, noise, v0, h0, n0):
    n_integrations = int(dic['integrations'])
    ## Input variables
    v_train = np.zeros(len(Iapps)*n_integrations)
    h_train = np.zeros(len(Iapps)*n_integrations)
    n_train = np.zeros(len(Iapps)*n_integrations)
    ## Targets
    target_v = np.zeros(len(Iapps)*n_integrations)
    target_h = np.zeros(len(Iapps)*n_integrations)
    target_n = np.zeros(len(Iapps)*n_integrations)
    ## Parameters
    h = dic['h']
    gK = dic['gK']
    gNa = dic['gNa']
    gL = dic['gL']
    vK = dic['vK']
    vNa = dic['vNa']
    vL = dic['vL']
    phi = dic['phi']
    Cm = dic['Cm']

    np.random.seed(1)
    noise_1 = np.random.uniform(0, noise, len(Iapps)*n_integrations)
    noise_2 = np.random.uniform(0, noise, len(Iapps)*n_integrations)
    gK = gK*(1+noise_1)
    gL = gL*(1+noise_2)
    
    ## Euler
    i=0
    for n,I in enumerate(Iapps):
        vn = np.zeros(n_integrations+1)
        hn = np.zeros(n_integrations+1)
        nn = np.zeros(n_integrations+1)
        vn[0] = v0
        hn[0] = h0
        nn[0] = n0
        for k in range(n_integrations):
            # leakage current (iL)
            iL = gL[n_integrations*n:n_integrations*(n+1)][k]*(v0-vL)
            #iL = gL*(v0-vL)
            # sodium current (iNa)
            am = -0.1*(v0+33.0)/(np.exp(-0.1*(v0+33.0))-1.0)
            bm = 4.0*np.exp(-(v0+58.0)/12.0)
            minf = am/(am+bm)
            iNa = gNa*(minf**3)*h0*(v0-vNa)
            ah = 0.07*np.exp(-(v0+50.0)/10.0)
            bh = 1/(1+np.exp(-0.1*(v0+20.0)))
            # delayed rectifier - potassium current (iK)
            iK = gK[n_integrations*n:n_integrations*(n+1)][k]*(n0**4)*(v0-vK)
            #iK = gK*(n0**4)*(v0-vK)
            an = -0.01*(v0+34.0)/(np.exp(-0.1*(v0+34.0))-1.0)
            bn = 0.125*np.exp(-(v0+44.0)/25.0)
            # Ionic currents (iIon)
            iIon = iNa+iK
            # ODE's system
            v1 = v0 + h * (-iL-iIon+I)/Cm
            h1 = h0 + h * (phi*(ah*(1-h0)-bh*h0))
            n1 = n0 + h * (phi*(an*(1-n0)-bn*n0))
            vn[k+1] = v1
            hn[k+1] = h1
            nn[k+1] = n1
            v0=v1
            h0=h1
            n0=n1
        for j in range(n_integrations):
            target_v[i+j] = vn[1:][j]
            target_h[i+j] = hn[1:][j]
            target_n[i+j] = nn[1:][j]
            v_train[i+j] = vn[:-1][j]
            h_train[i+j] = hn[:-1][j]
            n_train[i+j] = nn[:-1][j]
        i+=n_integrations    
    return (v_train, h_train, n_train), (target_v, target_h, target_n)

if __name__ == "__main__":
    
    ## Config
    param, jit_dict = config(sys.argv[0].split('.')[0])
    var = ['v', 'h', 'n']
    title_plots = 'Wang'
    
    if not param['only_evaluation']:
        training(param, jit_dict, Wang_3D, var, v0=param['v0_train'],
                                                h0=param['h0_train'],
                                                n0=param['n0_train'])
    evaluation(param, jit_dict, Wang_3D, var, title_plots, v0=param['v0_eval'],
                                                           h0=param['h0_eval'],
                                                           n0=param['n0_eval'])

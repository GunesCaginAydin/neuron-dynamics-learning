from .activation_functions import matrix_3D, matrix_2D
import pandas as pd, numpy as np, psutil, os

def matrix_G(param, inputs):
    if len(inputs) == 3:
        input_1, input_2, Iapp = inputs
        return matrix_2D(param, input_1, input_2, Iapp, hidden_layer(param, 2))
    if len(inputs) == 4:
        input_1, input_2, input_3, Iapp = inputs
        return matrix_3D(param, input_1, input_2, input_3, Iapp, hidden_layer(param, 3))

def hidden_layer(param, wn_dimension): #wn_dimension = 2 if 2D / 3 if 3D
    if param['bool_scale']: wavelons = 1
    else: wavelons = 0
    
    if wn_dimension == 2:
        if param['bool_lineal']: l = 2
        else: l = 0
        for m in range(param['resolution']+1):
            wavelons += (2**(3*m)+3*2**(2*m)+3*2**m)
        neurons = l+(param['n_sf']**3)*wavelons
        if (neurons**2)*param[param['dtype']]/2**30 > psutil.virtual_memory().total/(1024**3):
            print("- ERROR - Covariance matrix won't fit in memory. Unable to allocate", round((neurons**2)*param[param['dtype']]/2**30, 2), 'GB')
            exit()
        return neurons
        
    elif wn_dimension == 3:
        if param['bool_lineal']: l = 3
        else: l = 0
        for m in range(param['resolution']+1):
            wavelons += (2**(4*m)+4*2**(3*m)+6*2**(2*m)+4*2**m)
        neurons = l+(param['n_sf']**4)*wavelons
        if (neurons**2)*param[param['dtype']]/2**30 > psutil.virtual_memory().total/(1024**3):
            print("- ERROR - Covariance matrix won't fit in memory. Unable to allocate", round((neurons**2)*param[param['dtype']]/2**30, 2), 'GB')
            exit()
        return neurons

def generate_stepwise(param, n_Iapps, seeds):
    Iapps = np.linspace(param['I_min'], param['I_max'], n_Iapps)
    for s in seeds:
        np.random.RandomState(s).shuffle(Iapps)
    return Iapps

def normalize(param, train_data, Iapp):
    dnorm = {}
    for i,data in enumerate(train_data):
        dnorm['norm'+str(i)] = (data-param['min'+str(i)])/(param['max'+str(i)]-param['min'+str(i)])
    Inorm = (Iapp-param['I_min'])/(param['I_max']-param['I_min'])
    return tuple([norm for norm in dnorm.values()])+(Inorm,)

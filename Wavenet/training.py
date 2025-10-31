from .files import import_data, save_data
from .wavenet import hidden_layer, normalize, generate_stepwise
from .ooc_compute import ooc_training
from .user_interface import set_config
from dask import array as da
import numpy as np

def training(param, euler_dict, model, var, **IC_train):

    neurons = hidden_layer(param, len(var)) #compute num. nodes
    n_chunks, rows_chunk = set_config(param, neurons)
    
    seeds = [5061996, 5152017]
    if param['generateIapp']:
        Iapps = generate_stepwise(param, param['n_Iapp'], seeds)
        save_data('inputs.parquet', da.from_array(Iapps))
    else: Iapps = import_data(param)

    euler_dict['integrations'] = param['integrations']
    noise = 0 #euler_dict['noise'] #no noise at training
    train_data, target = model(euler_dict, Iapps, noise, **IC_train)
    Iapps = np.array([I for I in Iapps for n_times in range(param['integrations'])])
    
    # inputs = state_var_1, state_var_2, ..., Iapps
    inputs = normalize(param, train_data, Iapps) # inputs[-1] = normalized Iapps
    # tuples = [(input_1, target_1), (input_2, target_2) [...]]
    tuples = [(input_, target[i]) for i,input_ in enumerate(inputs[:-1])]
    
    ooc_training(param, var, inputs[-1], tuples, rows_chunk, n_chunks, neurons)

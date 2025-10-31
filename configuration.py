from numba.typed import Dict
from numba import types
import Wavenet.files as files, numpy as np

"""
Parameters
----------
dtype : str
    Defines the desidered precision float64 or float32.
float64 : int
    One bytes number.
float32 : int
    One bytes number.
regularizer_multiplier : float
    Multiplier $\mu$ from the regularizer $\mu x max_eigenvalue x ||\weights||²$.
fita_chunk_inf : float
    Minimum MB per FX matrix chunk. It should be reduced for PCs with low RAM.
fita_chunk_sup: float
    Minimum MB per FX matrix chunk.
recovery : bool
    If True training resumes from a saved matrix.zarr file.
recovery_FTF : bool
    If True training resumes from a saved FTF.parquet file.
recovery_var : int
    Resumes the training from the last computed weights. 0 if no weights are computed.
    1 if only the first state variable's weights are computed, and so on.
bool_lineal : bool
    If True identity neurons are used along the wavenet basis.
bool_scale : bool
    If True V_0 space is used in the wavenet basis.
generateIapp : bool
    If True Iapps are randomized for training, if False Iapps are imported from a .parquet file.
generateIapp_eval_oscillatory : bool
    If True oscillatory Iapps are used in the evaluation.
generate_Iapp_eval_stepwise : bool
    If True stepwise Iapps are used in the evaluation.
shuffle : bool
    If True imported Iapps are shuffled. Only if generate_Iapp_eval_* are both False
only_evaluation:
    If True only evaluation is executed from the computed weights in results_folder.
matrices_folder : str
    Directory to save temporal matrices.
matrix_folder : str
    Subdirectory from matrices_folder to save the matrix.zarr file.
results_folder : str
    Directory to save the weights and evaluation data.
cname : str
    Compression algorithm ('blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd').
clevel : int
    Level of compression. Ranges from 1 to 9. 
sec_factor : float
    If RAM is bussy, increase value to write lighter parallel files.
integrations : int
    Integrations per Iapp, to generate training data.
n_Iapp : int
    Number of Iapps, to generate training data.
I_max & I_min : float
    Boundary domain for Iapps in training.
resolution : int
    Level of resolution of the wavenet basis (-1, 0, 1, 2...).
n_sf : int
    Number of superposed functions (1, 2, 3, 4...).
fscale : str
    Scale function ('haar', 'hat', 'quadratic', 'bicubic').
integrations_eval : int
    Integrations per Iapp, for the evaluation.
n_Iapps_eval : int
    Number of Iapps, for the evaluation.
rmse_eval : int
    Number of integrations from where to compute the relative error, to avoid the transient. period.
I_min_eval & I_max_eval : float
    Boundary domain for Iapps in evaluation.

Distributed client parameters
-----------------------------
n_workers : int
    CPUs used for dask.distributed scheduler (even number or 1).
threads_per_worker : int
    Threads per worker.
memory_limit : int
    Memory per worker.
processes : bool
    If False it uses single-thread scheduler. If True, a multiprocessing scheduler.
client_temp_data : str
    Temporal client directory.
"""

param0 = {
    'dtype':'float64',
    'float64':8,
    'float32':4,
    
    'regularizer_multiplier': 2e-16,
    
    'fita_chunk_inf': 2000,
    'fita_chunk_sup': 10000,
    
    'recovery': False,
    'recovery_FTF': False,
    'recovery_var': 0,
    
    'bool_lineal': True,
    'bool_scale': True,          
    
    'generateIapp': True,
    'generateIapp_eval_oscillatory': False,
    'generateIapp_eval_stepwise': True,
    'shuffle': False,
    
    'only_evaluation': False,
    
    'matrix_folder': 'temporary_dir/FX',
    'matrices_folder': 'temporary_dir',
    'results_folder': 'results',
    
    'cname': 'zstd',
    'clevel': 1,
    
    'threads_per_worker': 1,
    #'memory_limit': '8G',
    'processes': True,
    #'client_temp_data': 'temporary_dir',
}

ML_config = {         
    'sec_factor': 1.7,
    'n_workers': 12,
    
    ## Training
    'integrations': 5000,
    'n_Iapp': 10000,
    'I_max': 100,
    'I_min': -10,
    'resolution': 0,
    'n_sf': 5,
    'fscale': 'bicubic',
    ## Normalization limits
    'max0': 0.55,
    'min0': -0.04, # w
    'max1': 40.,
    'min1': -70.,  # v    
    # Initial conditions
    'w0_train': 0.17,
    'v0_train': -50,
    
    ## Evaluation
    'integrations_eval': 2000,	#stepwise: 2000 || oscillatory: 1
    'n_Iapp_eval': 50,			#stepwise: 50   || oscillatory: 100000
    'rmse_eval': 2000,
    'I_max_eval': 60,
    'I_min_eval': 20,
    # Initial conditions
    'w0_eval': 0.25,
    'v0_eval': -40,
    
    **param0
}

FHN2D_config = {         
    'sec_factor': 1.7,
    'n_workers': 12,
    
    ## Training
    'integrations': 5000,
    'n_Iapp': 10000,
    'I_max': 0.1,
    'I_min': 0.0,
    'resolution': 0,
    'n_sf': 5,
    'fscale': 'bicubic',
    ## Normalization limits
    'max0': 0.02,
    'min0': -0.26, # w
    'max1': 0.8,
    'min1': -0.16,  # v    
    # Initial conditions
    'w0_train': -0.1,
    'v0_train': 0.28,
    
    ## Evaluation
    'integrations_eval': 2000,	#stepwise: 2000 || oscillatory: 1
    'n_Iapp_eval': 50,			#stepwise: 50   || oscillatory: 100000
    'rmse_eval': 2000,
    'I_max_eval': 0.09,
    'I_min_eval': 0.07,
    # Initial conditions
    'w0_eval': -0.11,
    'v0_eval': 0.3,
    
    **param0
}

FHN3D_config = {         
    'sec_factor': 2,
    'n_workers': 8,
    
    ## Training
    'integrations': 5000,
    'n_Iapp': 1500,
    'I_max': 0.1,
    'I_min': 0.0,
    'resolution': 0,
    'n_sf': 5,
    'fscale': 'bicubic',
    ## Normalization limits
    'max0': 0.02,
    'min0': -0.3, # w
    'max1': 0.9,
    'min1': -0.2, # v
    'max2': -0.75,
    'min2': -1.5, # y
    # Initial conditions
    'w0_train': -0.1,
    'v0_train': 0.0,
    'y0_train': -1.0,
    
    ## Evaluation
    'integrations_eval': 2000,	#stepwise: 2000 || oscillatory: 1
    'n_Iapp_eval': 50,			#stepwise: 50   || oscillatory: 100000
    'rmse_eval': 2000,
    'I_max_eval': 0.07,
    'I_min_eval': 0.05,
    # Initial conditions
    'w0_eval': -0.11,
    'v0_eval': 0.1,
    'y0_eval': -0.9,
    
    **param0
}

Wang_config = {         
    'sec_factor': 4,
    'n_workers': 1,
    
    ## Training
    'integrations': 4000,
    'n_Iapp': 11500,
    'I_max': 5,
    'I_min': -5,
    'resolution': 1,
    'n_sf': 4,
    'fscale': 'quadratic',
    ## Normalization limits
    'max0': 60,
    'min0': -120, # v
    'max1': 1.1,
    'min1': -0.1, # h
    'max2': 1.1,
    'min2': -0.1, # n   
    # Initial conditions
    'v0_train': -55.0,
    'h0_train': 0.83,
    'n0_train': 0.11,
    
    ## Evaluation
    'integrations_eval': 20000,	#stepwise: 20000 || oscillatory: 1
    'n_Iapp_eval': 50,			#stepwise: 50    || oscillatory: 1000000
    'rmse_eval': 20000,
    'I_max_eval': 3,
    'I_min_eval': -3,
    # Initial conditions
    'v0_eval': -50.0,
    'h0_eval': 0.8,
    'n0_eval': 0.1,
    
    **param0
}

################################################################################################

if param0['generateIapp_eval_oscillatory'] == True \
    and param0['generateIapp_eval_stepwise'] == False:
    dict0 = {'noise': 0.0}
elif param0['generateIapp_eval_stepwise'] == True \
    and param0['generateIapp_eval_oscillatory'] == False:
    dict0 = {'noise': 0.01}    
elif param0['generateIapp_eval_stepwise'] == True \
    and param0['generateIapp_eval_oscillatory'] == True:
    print('\n', '- ERROR - The evaluation analysis is not properly set up')

ML_jit = {
    # Euler parameters
    'h': 0.05,
    'phi': 1/15,
    'gL': 2.0,
    'gK': 8.0,
    'Cm': 20.0,
    'gCa': 4.0,
    'EL': -60,
    'EK': -84,
    'ECa': 120,
    'V1': -1.2,
    'V2': 18,
    'V3': 12,
    'V4': 17.4,
    
    # Parameters for the oscillatory Iapp signal
    #I1 = I0 + h * ((x0 + mu_est*np.cos(w*t) - I0)/tau + sigma*white_noise[t])
    'w': (2*np.pi/100)/(ML_config['integrations_eval']*ML_config['n_Iapp_eval'])*3000,
    'tau': 10,
    'mu_est': 10.0,
    'sigma': 9.5,
    
    **dict0
}

FHN2D_jit = {
    # Euler parameters
    'h': 0.05,
    'a': 0.14,
    'gamma': 2.54,
    'eps': 0.1,
    
    # Parameters for the oscillatory Iapp signal
    'w': (2*np.pi/100)/(FHN2D_config['integrations_eval']*FHN2D_config['n_Iapp_eval'])*3000,
    'tau': 10,
    'mu_est': 0.005,
    'sigma': 0.0045,
    
    **dict0
}

FHN3D_jit = {
    # Euler parameters
    'h': 0.05,
    'a': 0.14,
    'gamma': 2.54,
    'eps': 0.1,
    'alpha': 0.02,
    'c': -0.775,
    'd': 1,
    'mu': 0.01,
    
    # Parameters for the oscillatory Iapp signal
    'w': (2*np.pi/100)/(FHN3D_config['integrations_eval']*FHN3D_config['n_Iapp_eval'])*3000,
    'tau': 10,
    'mu_est': 0.005,
    'sigma': 0.0045,
    
    **dict0
}

Wang_jit = {
    # Euler parameters
    'h': 0.005,
    'gNa': 45.0,
    'vK': -80.0,
    'vNa': 55.0,
    'vL': -65.0,
    'phi': 4.0,
    'gL': 0.1,
    'gK': 18.0,
    'Cm': 1.0,
    
    # Parameters for the oscillatory Iapp signal
    'w': (2*np.pi/100)/(Wang_config['integrations_eval']*Wang_config['n_Iapp_eval'])*3000,
    'tau': 10,
    'mu_est': 1.5,
    'sigma': 2.85,
    
    **dict0
}

################################################################################################

def dict_for_jit(d):
    # Dict with keys as string and values of type float
    jit_dic = Dict.empty(key_type=types.unicode_type, value_type=types.float64)    
    for key,value in d.items():
      jit_dic[key] = value
    return jit_dic

def config(model):    
    models_dict = {'Morris_Lecar': (ML_config, ML_jit),
                   'Nagumo_2D': (FHN2D_config, FHN2D_jit),
                   'Nagumo_3D': (FHN3D_config, FHN3D_jit),
                   'Wang_3D': (Wang_config, Wang_jit)}
    d1, d2 = models_dict[model]
    ## Setup files first
    files.set_files(d1)
    return d1, dict_for_jit(d2)

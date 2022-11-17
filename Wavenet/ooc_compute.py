import pandas as pd, numpy as np, psutil, zarr, dask, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dask.distributed import Client, LocalCluster
from dask import dataframe as dd, array as da
from dask.distributed import progress
from itertools import combinations
from numpy.linalg import lstsq
from scipy.linalg import eigh
from numcodecs import Blosc
from tqdm import tqdm

from .files import read_data, save_data
from .wavenet import matrix_Fx
import multiprocessing.popen_spawn_posix #in windows: import multiprocessing.popen_spawn_win32

def factorial_decomposition(num):
    factor = 2
    l = []
    while factor*factor <= num:
        if num % factor:
            factor += 1
        else:
            num = num//factor
            l.append(factor)
    if num > 1:
        l.append(num)
    return l

def prod_func(n):
    p = n[0]
    for elem in n[1:]:
        p *= elem
    return p

def divisors(l):
    # n-1 lowest dividors of n_chunks: int
    n_factors = np.arange(1, len(l))
    ll = []
    for n in n_factors:
        comb_factors = list(combinations(l, n))
        divisors = list(map(prod_func, comb_factors))
        ll.append(divisors)
    div_list = list(set([e for elem in ll for e in elem]))
    return div_list

def cpu_handling(param, n_chunks, rows_chunk, neurons):
    div_list = [1]+divisors(factorial_decomposition(n_chunks)) 
    div_list.reverse()
    cpu_compensator = 0 #number of CPUs not used to compensate for the RAM per CPU needed
    while cpu_compensator < mp.cpu_count():
        """
        It first tries to handle as much chunks using as much CPUs as possible
        If all chunks asigned to the CPUs are too much for the RAM it then reduces the CPUs
        """
        for elem in div_list:
            if rows_chunk*elem*neurons*8 <= psutil.virtual_memory().available/(mp.cpu_count()-cpu_compensator)/param['sec_factor']:
                print('Minus:', cpu_compensator, 'CPUs | Chunk/CPU:',
                      round(rows_chunk*elem*neurons*8/2**30,2), 'GB')
                return elem, cpu_compensator
        cpu_compensator += 1
    print('- ERROR - The sub-chunk is too big, not enough space'); exit()

def client_distributed(param):
    # Distributed client config
    worker_kwargs = {'processes': param['processes'],
                     'n_workers': param['n_workers'],
                     'threads_per_worker': param['threads_per_worker'],
                     'silence_logs': 40,
                     #'memory_limit': param['memory_limit'], #per worker
                     'memory_target_fraction': 0.95,
                     'memory_spill_fraction': 0.99,
                     'memory_pause_fraction': False,
                     #'local_dir': param['client_temp_data']
                     }

    dask.config.set({"distributed.worker.memory.terminate": False})
    # Setup Dask distributed client
    cluster = LocalCluster(**worker_kwargs)
    return Client(cluster)

def parallel_writes(z, j, nf, param, Iapps, tuples):
    inputs = [input_[0][nf*j:nf*(j+1)] for input_ in tuples]+[Iapps[nf*j:nf*(j+1)]]
    z[nf*j:nf*(j+1)] = matrix_Fx(param, inputs)
    return j #the return only tracks the processes
    
def ooc_training(param, var, Iapps, tuples, rows_chunk, n_chunks, neurons):
    
    factor, cpu_compensator = cpu_handling(param, n_chunks, rows_chunk, neurons)
    nf = rows_chunk*factor
    n_chunks //= factor # 1 dataset/chunk
    
    if not param['recovery']:
        ## THE .zarr MATRIX IS CONSTRUCTED ##
        compressor = Blosc(cname=param['cname'], clevel=param['clevel'])#, shuffle=Blosc.SHUFFLE)
        #synchronizer = zarr.ProcessSynchronizer('temp.sync')
        synchronizer = zarr.sync.ThreadSynchronizer()
        f = zarr.DirectoryStore(param['matrix_folder']+'/matriu.zarr')
        z = zarr.create(store = f, shape=(Iapps.shape[0], neurons),
                        overwrite = True, compressor = compressor,
                        synchronizer = synchronizer, dtype = param['dtype'])
        cont = 0
        with ProcessPoolExecutor(mp.cpu_count()-cpu_compensator) as ex:
            futures = [ex.submit(parallel_writes, z, j, nf, param, Iapps, tuples) for j in range(n_chunks)]
            for job in tqdm(as_completed(futures), total=len(futures),
                            desc='Saving matrix', unit='chunk', leave=True):
                cont += job.result()
                
        if cont != np.sum([i for i in range(n_chunks)]): exit()
    
    
    client = client_distributed(param)    
    f = zarr.open(param['matrix_folder']+'/matriu.zarr', 'r')
    FX = da.from_array(f, chunks=(rows_chunk, neurons))
    if not param['recovery_FTF']:
        print('--- Saving matrix FTF ---')
        FTF = dd.from_dask_array(da.dot(FX.T, FX), columns = [str(elem) for elem in np.arange(neurons)])
        FTF = FTF.persist()
        progress(FTF) #progress bar
        FTF.to_parquet(param['matrices_folder']+'/FTF.parquet')
        
    FTF = np.array(pd.read_parquet(param['matrices_folder']+'/FTF.parquet'))
    vaps = eigh(FTF, eigvals_only=True)
    FTF = FTF + np.identity(FTF.shape[1])*param['regularizer_multiplier']*vaps[-1].real
    
    print('\n Eigh_max =', vaps[-1])
    print('Regularizer parameter:', param['regularizer_multiplier']*vaps[-1].real, '\n')    
    
    client.close() #close the client
    client.shutdown() #close the scheduler
    param['n_workers'] = 4
    
    for i in range(param['recovery_var'], len(tuples)):
        client = client_distributed(param)
        print('\n'+'--- Training', var[i], '---')
        target = tuples[i][1].reshape(len(tuples[i][1]), 1)
        ooc_loss(param, FX, FTF, target, rows_chunk, var[i])
        client.close()
        client.shutdown()
    print('\n')

def ooc_loss(param, FX, FTF, target, rows_chunk, var):
    
    Y = da.from_array(target, chunks = (rows_chunk, 1))    
    FTY = dd.from_dask_array(da.dot(FX.T, Y), columns = ['0'])
    FTY = FTY.persist()
    progress(FTY)
    FTY.to_parquet(param['matrices_folder']+'/FTY.parquet')
    b = np.array(pd.read_parquet(param['matrices_folder']+'/FTY.parquet'))
    
    _, s, _ = np.linalg.svd(FTF)
    weights, _, rank, _ = lstsq(FTF, b, rcond=s.min()/s.max()*(1-np.finfo('float64').eps))
    weights = da.from_array(weights, chunks = (rows_chunk, 1))    
    save_data(param['results_folder']+'/weights_' + var + '.parquet', weights)
    
    #print('->', var, 'MSE at level =', param['resolution']+2, 'is:', ooc_mse(FX,target,weights))

def ooc_mse(FX, target, weights):
    save_data(param['matrix_folder']+'/temp.parquet', da.dot(FX, weights))   
    FW = read_data(param['matrices_folder']+'/temp.parquet')
    return np.sum((target - FW)**2)/len(target)

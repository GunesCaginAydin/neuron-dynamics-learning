import pandas as pd, numpy as np, os, shutil
from dask import dataframe as dd, array as da

def set_files(param):
    if not param['recovery']:
        main_path = os.getcwd()
        path = os.path.join(main_path, param['matrices_folder'])
        if os.path.exists(path): shutil.rmtree(path)
        os.mkdir(path)
        os.mkdir(os.path.join(main_path, param['matrix_folder']))
    if not (param['only_evaluation']  or param['recovery_var'] != 0):
        main_path = os.getcwd()
        path = os.path.join(main_path, param['results_folder'])
        if os.path.exists(path): shutil.rmtree(path)
        os.mkdir(path)

def import_data(param):
    return read_data('inputs.parquet') #Iapps

def save_data(f, vector):
    dd.from_dask_array(vector, columns = ['0']).to_parquet(f)

def read_data(f):
    # data structure f = [[elem 1],[elem 2],[...],[elem n]]
    # return np.array([n elem])
    return np.array(pd.read_parquet(f)).T[0]

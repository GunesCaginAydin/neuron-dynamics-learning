from terminaltables import SingleTable
import psutil

# Compression rations according to the lvl of resultion. Experty values.
ratio_comp = {-1:3, 0:6, 1:15, 2:15, 3:30}

def display_configuration(param, neurons, n_chunks):
    nbytes = param['integrations']*param['n_Iapp']*neurons*param[param['dtype']]
    config = [['n_chunks', n_chunks,
               'rows_chunk', param['n_Iapp']*param['integrations']//n_chunks],
              ['n_integrations', param['integrations'],
               'n_Iapps', param['n_Iapp']],
              ['fita_chunk_inf', param['fita_chunk_inf'],
               'fita_chunk_sup', param['fita_chunk_sup']],
              ['Raw memory G(x)', str(round(nbytes/(2**30), 2)) + ' GB',
               'Memory/chunk', str(round(nbytes/n_chunks/(2**20),2)) + ' MB']]
    
    disp = SingleTable(config, ' WAVENET WITH '+ str(neurons) +' NEURONS '+'--> OUT-OF-CORE ')
    disp.inner_row_border = True
    disp.justify_columns = {0: 'center', 1: 'center', 2: 'center', 3: 'center'}
    return disp.table
    
def change_parametrization(param, neurons, marker):
    #int: marker to know what have to changed
    n_chunks = chunk_size(param, neurons, marker)
    print('')
    print(display_configuration(param, neurons, n_chunks))
    print('Change n_Iapps [i], nº of integrations [p], fita_chunk_inf [inf] or fita_chunk_sup [sup]?')
    answer = input("(i/p/inf/sup): ")
    if answer == 'i':
        new_Iapps = input('Enter new n_Iapp value: ')
        param['n_Iapp'] = int(new_Iapps) #overwrite param
        print('\n'*100)
        return set_config(param, neurons, 0)
    elif answer == 'p':
        new_punts = input('Enter new n_integrations value: ')
        param['integrations'] = int(new_punts)
        print('\n'*100)
        return set_config(param, neurons, 1)
    elif answer == 'inf':
        new_fita_inf = input('Enter new fita_inf value [MB]: ')
        param['fita_chunk_inf'] = int(new_fita_inf)
        print('\n'*100)
        return set_config(param, neurons, marker)
    elif answer == 'sup':
        new_fita_sup = input('Enter new fita_sup value [MB]: ')
        param['fita_chunk_sup'] = int(new_fita_sup)
        print('\n'*100)
        return set_config(param, neurons, marker)
    print('')
    return n_chunks, param['n_Iapp']*param['integrations']//n_chunks    

def set_config(param, neurons, marker = 0):
    memory = param['integrations']*param['n_Iapp']*neurons*param[param['dtype']]/(2**30)
    #+10% disk margin for the distributed temporary_directory
    if (memory <= psutil.disk_usage('/')[2]/(1024**3)*ratio_comp[param['resolution']]/1.1) or param['recovery']:
        n_chunks, rows_chunk = change_parametrization(param, neurons, marker)
    else:
        print('\n','- ERROR - Available disk space:', round(psutil.disk_usage('/')[2]/(2**30), 2), 'GB', 'Required disk space:', round(memory/ratio_comp[param['resolution']]*1.1, 2), 'GB')
        exit()
    return n_chunks, rows_chunk
    
def chunk_size(param, neurons, marker):
    matrix_MB = param['integrations']*param['n_Iapp']*neurons*param[param['dtype']]/(2**20)
    n_chunks_max = int(matrix_MB/param['fita_chunk_inf']+1)
    # n_chunks_max is the maximum num. of chunks of minimum rows
    trigger = 0
    while n_chunks_max > matrix_MB/param['fita_chunk_sup']:
        rows_chunk = param['integrations']*param['n_Iapp']/n_chunks_max
        if rows_chunk % 1 == 0 and matrix_MB/n_chunks_max > param['fita_chunk_inf']:
            trigger = 1
            break
        n_chunks_max -= 1
    if trigger == 1: return n_chunks_max
    else:
        print('- ERROR - No possible configuration.')
        if marker == 0:
            new_Iapps = input('Enter new n_Iapp value: ')
            param['n_Iapp'] = int(new_Iapps)
            return chunk_size(param, neurons, marker)
        elif marker == 1:
            new_punts = input('Enter new n_integrations value: ')
            param['integrations'] = int(new_punts)
            return chunk_size(param, neurons, marker)

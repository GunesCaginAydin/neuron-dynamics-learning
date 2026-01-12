def bin2dict(bin_spikes_matrix : torch.Tensor) -> dict:
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds,spike_times):
        if row_ind in row_inds_spike_times_map.keys():
            row_inds_spike_times_map[row_ind].append(syn_time)
        else:
            row_inds_spike_times_map[row_ind] = [syn_time]
    return row_inds_spike_times_map

def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind,spike_time] = 1.0
    
    return bin_spikes_matrix

def parse_sim_experiment_file_with_DVT(sim_experiment_file, DVT_PCA_model=None, print_logs=False):
    if print_logs:
        print('-----------------------------------------------------------------')
        print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
        loading_start_time = time.time()
        
    if sys.version_info[0]<3:
        experiment_dict = pickle.load(open(sim_experiment_file, "rb" ))
    else:
        experiment_dict = pickle.load(open(sim_experiment_file, "rb" ),encoding='latin1')
    
    # gather params
    num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
    num_segments    = len(experiment_dict['Params']['allSegmentsType'])
    sim_duration_ms = experiment_dict['Params']['totalSimDurationInSec'] * 1000
    num_ex_synapses  = num_segments
    num_inh_synapses = num_segments
    num_synapses = num_ex_synapses + num_inh_synapses
    
    # collect X, y_spike, y_soma
    X = np.zeros((num_synapses,sim_duration_ms,num_simulations), dtype='bool')
    y_spike = np.zeros((sim_duration_ms,num_simulations))
    y_soma  = np.zeros((sim_duration_ms,num_simulations))
    
    # if we recive PCA model of DVTs, then output the projection on that model, else return the full DVTs
    if DVT_PCA_model is not None:
        num_components = DVT_PCA_model.n_components
        y_DVTs  = np.zeros((num_components,sim_duration_ms,num_simulations), dtype=np.float32)
    else:
        y_DVTs  = np.zeros((num_segments,sim_duration_ms,num_simulations), dtype=np.float16)
    
    # go over all simulations in the experiment and collect their results
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex  = dict2bin(sim_dict['exInputSpikeTimes'] , num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
        X[:,:,k] = np.vstack((X_ex,X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times,k] = 1.0
        y_soma[:,k] = sim_dict['somaVoltageLowRes']
        
        # if we recive PCA model of DVTs, then output the projection on that model, else return the full DVTs
        curr_DVTs = sim_dict['dendriticVoltagesLowRes']
        # clip the DVTs (to mainly reflect synaptic input and NMDA spikes (battery ~0mV) and diminish importance of bAP and calcium spikes)
        curr_DVTs[curr_DVTs > 2.0] = 2.0
        if DVT_PCA_model is not None:
            y_DVTs[:,:,k] = DVT_PCA_model.transform(curr_DVTs.T).T
        else:
            y_DVTs[:,:,k] = curr_DVTs
        
    if print_logs:
        loading_duration_sec = time.time() - loading_start_time
        print('loading took %.3f seconds' %(loading_duration_sec))
        print('-----------------------------------------------------------------')

    return X, y_spike, y_soma, y_DVTs


def parse_multiple_sim_experiment_files_with_DVT(sim_experiment_files, DVT_PCA_model=None):
    
    for k, sim_experiment_file in enumerate(sim_experiment_files):
        X_curr, y_spike_curr, y_soma_curr, y_DVT_curr = parse_sim_experiment_file_with_DVT(sim_experiment_file, DVT_PCA_model=DVT_PCA_model)
        
        if k == 0:
            X       = X_curr
            y_spike = y_spike_curr
            y_soma  = y_soma_curr
            y_DVT   = y_DVT_curr
        else:
            X       = np.dstack((X,X_curr))
            y_spike = np.hstack((y_spike,y_spike_curr))
            y_soma  = np.hstack((y_soma,y_soma_curr))
            y_DVT   = np.dstack((y_DVT,y_DVT_curr))

    return X, y_spike, y_soma, y_DVT


def parse_sim_experiment_file(sim_experiment_file):
    
    print('-----------------------------------------------------------------')
    print("loading file: '" + sim_experiment_file.split("\\")[-1] + "'")
    loading_start_time = time.time()
    experiment_dict = pickle.load(open(sim_experiment_file, "rb" ))
    
    # gather params
    num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
    num_segments    = len(experiment_dict['Params']['allSegmentsType'])
    sim_duration_ms = experiment_dict['Params']['totalSimDurationInSec'] * 1000
    num_ex_synapses  = num_segments
    num_inh_synapses = num_segments
    num_synapses = num_ex_synapses + num_inh_synapses
    
    # collect X, y_spike, y_soma
    X = np.zeros((num_synapses,sim_duration_ms,num_simulations), dtype='bool')
    y_spike = np.zeros((sim_duration_ms,num_simulations))
    y_soma  = np.zeros((sim_duration_ms,num_simulations))
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex  = dict2bin(sim_dict['exInputSpikeTimes'] , num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
        X[:,:,k] = np.vstack((X_ex,X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times,k] = 1.0
        y_soma[:,k] = sim_dict['somaVoltageLowRes']

    loading_duration_sec = time.time() - loading_start_time
    print('loading took %.3f seconds' %(loading_duration_sec))
    print('-----------------------------------------------------------------')

    return X, y_spike, y_soma


def parse_multiple_sim_experiment_files(sim_experiment_files):
    
    for k, sim_experiment_file in enumerate(sim_experiment_files):
        X_curr, y_spike_curr, y_soma_curr = parse_sim_experiment_file(sim_experiment_file)
        
        if k == 0:
            X       = X_curr
            y_spike = y_spike_curr
            y_soma  = y_soma_curr
        else:
            X       = np.dstack((X,X_curr))
            y_spike = np.hstack((y_spike,y_spike_curr))
            y_soma  = np.hstack((y_soma,y_soma_curr))

    return X, y_spike, y_soma

import matplotlib.pyplot as plt, matplotlib.pylab as pylab, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .files import read_data

# Graphic's format
pylab.rcParams['axes.labelsize'] = 9
pylab.rcParams['xtick.labelsize'] = 9
pylab.rcParams['ytick.labelsize'] = 9
pylab.rcParams['legend.fontsize'] = 9
pylab.rcParams['font.family'] = 'serif'
pylab.rcParams['font.sans-serif'] = ['Computer Modern Roman']
pylab.rcParams['text.usetex'] = True
pylab.rcParams['figure.figsize'] = 7.3, 4.2

def plotting(param, title, var, Iapps, targets, h):
    d_var={}
    time = np.array([j for j in range(len(targets[0]))])*h
    for i,target in enumerate(targets):
        plt.figure()
        plt.title(title + ' ' + var[i] + ' output')
        plt.xlabel('Time (ms)')
        plt.ylabel(var[i])
        d_var['pred_'+str(i)] = read_data(param['results_folder']+'/predicted_'+var[i]+'.parquet')
        plt.plot(time, target, label='Target', color='blue', linestyle='-', lw=0.45)
        plt.plot(time, d_var['pred_'+str(i)], label='WNN', color='orange', linestyle='-', lw=0.4)
        plt.legend(loc = 1)
        plt.savefig('Outputs_'+var[i]+'.png', bbox_inches='tight', dpi=300)
    
    plt.figure()
    plt.title('Randomised forced term')
    plt.xlabel('Time (ms)')
    plt.ylabel('Iapp')
    plt.plot(time, Iapps, color='blue', marker=',', linestyle='-', lw=0.5)
    plt.savefig('Iapps.png', bbox_inches='tight', dpi=300)
    
    # Phase portrait 2D
    if len(var) == 2:
        d_var['pred_2'] = Iapps
        targets  = targets+(Iapps,)
        var = var+['Iapp']
        
        plt.figure()
        plt.title(title + ' phase portrait')
        plt.plot(targets[0], targets[1], label='Target', color='blue', linestyle='-', lw=0.45)
        plt.plot(d_var['pred_0'], d_var['pred_1'], label='WNN', color='orange', linestyle='-', lw=0.4)
        plt.xlabel(var[0])
        plt.ylabel(var[1])
        plt.legend(loc = 1)
        plt.savefig('Phase_portrait_2D.png', bbox_inches='tight', dpi=300)
        
    # Phase portrait 3D
    phase_portrait(title, var, [d_var['pred_'+str(i)] for i in range(len(var))], targets)
    plt.show()

def phase_portrait(title, var, outputs, targets):
    predict_1, predict_2, predict_3 = outputs
    tar_1, tar_2, tar_3 = targets
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(title + ' phase portrait')
    ax.plot(tar_1, tar_3, tar_2, label='Target', color='blue', linestyle='dotted', lw = 0.45)
    ax.plot(predict_1, predict_3, predict_2, label='WNN', color='orange', linestyle='dotted', lw = 0.4)
    ax.set_xlabel(var[0])
    ax.set_ylabel(var[2])
    ax.set_zlabel(var[1])
    ax.legend()


import numpy as np
import DA
from Filters import Seik, Ghosh, GhoshV1, GhoshEighT

import matplotlib.pyplot as plt
import numpy as np

import gc

#def plot_results(arrays=None, xtitles=[0.1,0.2,0.3], xticks=[str(x) for x in 2**np.arange(3,7)-1], yticks=[str(x) for x in [0.8,0.9,0.95,1.0]], titles=['MSRE_total','MSRE_observed','MSRE_unobserved']):
def plot_results(arrays=None, truth_t_array=None, delta_obs_array=None, forget_array=None, EnsSize_array=None, titles=['RMSE ratio','RMSE ratio (only observed vars)','RMSE ratio (only unobserved vars)'], cmap='YlGn_r'):
    
    if arrays is None:
        with np.load('Z.npz') as f:
            arrays={}
            for key in f.files:
                arrays[key]=f[key]
                
    if truth_t_array is None:
        truth_t_index=np.arange(len(arrays['truth_t_array']))
    else:
        truth_t_index=np.array([i for i, val in enumerate(arrays['truth_t_array']) if val in truth_t_array])
    if delta_obs_array is None:
        delta_obs_index=np.arange(len(arrays['delta_obs_array']))
    else:
        delta_obs_index=np.array([i for i, val in enumerate(arrays['delta_obs_array']) if val in delta_obs_array])
    if forget_array is None:
        forget_index=np.arange(len(arrays['forget_array']))
    else:
        forget_index=np.array([i for i, val in enumerate(arrays['forget_array']) if val in forget_array])
    if EnsSize_array is None:
        EnsSize_index=np.arange(len(arrays['EnsSize_array']))
    else:
        EnsSize_index=np.array([i for i, val in enumerate(arrays['EnsSize_array']) if val in EnsSize_array])
    
    xtitles=arrays['delta_obs_array'][delta_obs_index]
    xticks=[str(x) for x in arrays['EnsSize_array'][EnsSize_index]]
    yticks=[str(x) for x in arrays['forget_array'][forget_index]]
    
    for metric, title in enumerate(titles):
        Z=arrays['Z'][truth_t_index[:,None,None,None],delta_obs_index[:,None,None],forget_index[:,None], EnsSize_index, metric]
        vmin=Z.min()
        vmax=Z.max()
        
        fig, axs = plt.subplots(Z.shape[0], Z.shape[1], sharex=True, sharey=True, figsize=(8,8))
        im=[[axs[i,j].imshow(Z[i,j], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower') for j in range(Z.shape[1])] for i in range(Z.shape[0])]
        
        for j, xtitle in enumerate(arrays['delta_obs_array'][delta_obs_index]):
            axs[0,j].set_title(f'{xtitle}')
            axs[-1,j].set_xlabel('EnsSize')
            axs[-1,j].set_xticks([float(x) for x in range(len(xticks))])
            axs[-1,j].set_xticklabels(xticks, rotation=45)
        for ax in axs[:, 0]:
            ax.set_ylabel('forget')
            ax.set_yticks([float(x) for x in range(len(yticks))])
            ax.set_yticklabels(yticks)
        for ax in axs.flatten():
            ax.tick_params(length=0.0)
            
        #fig.set_tight_layout(True)
        
        axs[0, len(delta_obs_index)//2].annotate('Obs Frequency', (0.5, 1), xytext=(0, 20),
                            textcoords='offset points', xycoords='axes fraction',
                            ha='center', va='bottom', size=14)
        axs[len(truth_t_index)//2, 0].annotate('Different random truths', (0, 0.5), xytext=(-40, 0),
                            textcoords='offset points', xycoords='axes fraction',
                            ha='right', va='center', size=14, rotation=90) 
        #fig.subplots_adjust(bottom=0.05, right=0.95)
        
        fig.colorbar(im[0][0], ax=axs)
        fig.suptitle(title,fontsize=22)
        
        fig.savefig('SAVED/'+title.replace(' ','_'))
    
    plt.show()
    
# make data
#X, Y = np.meshgrid(np.linspace(-3, 3, 16), np.linspace(-3, 3, 16))
#Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
#Z=X*Y

#plot_results(Z)

#exit()
if __name__=='__main__':
    N=2**6-2
    t_span=[0.0,20.0]
    obs_std=1.0
    n_experiments=100
    error_std=np.ones(N)*5.0
    model=DA.Lorentz96(F=8)

    obs_each=2
    indices=range(0,N,obs_each)
    obs=DA.ObsByIndex(np.zeros(N//obs_each),np.ones(N//obs_each)*obs_std, indices=indices)
    metrics=[
            DA.HalfTimeMean(DA.RmpeByTime(index= None), name='RmseTot'),
            DA.HalfTimeMean(DA.RmpeByTime(index= indices), name='RmseObserved'),
            DA.HalfTimeMean(DA.RmpeByTime(index= tuple(set(range(N))-set(indices))), name='RmseNotObserved'),
            ]

    #truth_t_array, delta_obs_array, forget_array, EnsSize_array = np.meshgrid([20.0,40.0,60.0], np.linspace(0.1, 0.3, 3), np.linspace(0.7, 1.0, 4), 2**np.arange(3,7)-1, indexing='ij')
    #for truth_t, delta_obs, forget, EnsSize in zip(truth_t_array.flatten(), delta_obs_array.flatten(), forget_array.flatten(), EnsSize_array.flatten()):
    
    try:
        with np.load('Z.npz') as f:
            saved={}
            for key in f.files:
                saved[key]=f[key]
    except Exception as err:
        print(f'impossible to load (error: {err}). Starting from scratch.')
        saved={'Z':np.array([]), 
                'truth_t_array':np.array([]),
                'delta_obs_array':np.array([]),
                'forget_array':np.array([]),
                'EnsSize_array':np.array([], dtype=np.int64)}
    
    Z=[]
    truth_t_array=[20.0, 40.0, 60.0, 80.0]
    delta_obs_array=[0.1, 0.15, 0.2, 0.25, 0.3]
    forget_array=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    EnsSize_array = list(2**np.arange(3,7)-1)
    Z=list(np.load('Z.bkp.npy'))
    for truth_t in truth_t_array[1:]:
        IC_truth,_ =model([0.0,truth_t],[0.01]+[0.0]*(N-1))
        del _
        gc.collect()
        for delta_obs in delta_obs_array:
            for forget in forget_array:
                for EnsSize in EnsSize_array:
                    print(f'computing for truth_t, delta_obs, forget, EnsSize = {(truth_t, delta_obs, forget, EnsSize)}')
                    
                    #if not all(truth_t==)
                    if all([truth_t in saved['truth_t_array'], delta_obs in saved['delta_obs_array'], forget in saved['forget_array'], EnsSize in saved['EnsSize_array']]):
                        Z.extend(list(saved['Z'][np.nonzero(saved['truth_t_array']==truth_t)[0][0],
                                                 np.nonzero(saved['delta_obs_array']==delta_obs)[0][0],
                                                 np.nonzero(saved['forget_array']==forget)[0][0],
                                                 np.nonzero(saved['EnsSize_array']==EnsSize)[0][0]]))
                        print('read from saved file')
                        continue
                    
                    ens_filters=[
                                Seik(EnsSize, forget=forget),
                                Ghosh(EnsSize, order=5, forget=forget),
                                ]
                    test=DA.TwinExperiment(t_span, model, ens_filters, metrics=metrics)
                    test.build_truth(IC_truth, delta=delta_obs)
                    test.build_obs(np.arange(t_span[0]+delta_obs,t_span[1],delta_obs), obs)
                    test.build_tests()
                    test.build_ICs(error_std, n_experiments=n_experiments)

                    test.run()
                    
                    metric_array=np.array([[metric.result[0].item() for metric in filter_test.metrics_result] for filter_test in test.tests])
                    Z.extend(list(metric_array[1]/metric_array[0]))
                    
                    del ens_filters
                    del test
                    del metric_array
                    gc.collect()
        del IC_truth
        gc.collect()
        np.save('Z.bkp', Z)
        
    Z=np.reshape(Z, [len(truth_t_array),len(delta_obs_array),len(forget_array),len(EnsSize_array),3])
    print('saving Z')
    np.savez('Z.npz', Z=Z, truth_t_array=truth_t_array, delta_obs_array=delta_obs_array, forget_array=forget_array, EnsSize_array=EnsSize_array )
    print('Z.npz saved')
            
    #plot_results(Z)



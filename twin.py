
import numpy as np
import DA
from Filters import Seik, Ghosh, GhoshV1, GhoshEighT

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

import gc
import os

#def plot_results(arrays=None, xtitles=[0.1,0.2,0.3], xticks=[str(x) for x in 2**np.arange(3,7)-1], yticks=[str(x) for x in [0.8,0.9,0.95,1.0]], titles=['MSRE_total','MSRE_observed','MSRE_unobserved']):
def plot_results(arrays=None, truth_t_array=None, delta_obs_array=None, forget_array=None, EnsSize_array=None, titles=['Total RMSE','Observed RMSE','Unobserved RMSE'], cmaps=['seismic_r','YlGn_r'], save=False, folder='SAVED'):
    
    if save:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
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
    yticks=[str(x) for x in arrays['forget_array'][forget_index]]+['best']
    
    #arrays['Z']=np.concatenate((arrays['Z'],arrays['Z'].min(axis=2, keepdims=True)),axis=2)
    arrays['Z']=np.concatenate((arrays['Z'],
                                np.take_along_axis(arrays['Z'], np.expand_dims(arrays['Z'][...,0:1].argmin(axis=2), axis=2), axis=2)),
                                axis=2)
    forget_index=np.concatenate((forget_index,[-1]))
    
    arrays['Z']=np.concatenate((arrays['Z'],np.sqrt((arrays['Z']**2).mean(axis=0, keepdims=True))),axis=0)
    #truth_t_index=np.concatenate((truth_t_index,[-1]))
    
    for nfilter, filtername in enumerate(['SEIK','GHOSH','ratio',]):
        for metric, title0 in enumerate(titles):
            title=title0+' '+filtername
            Z=arrays['Z'][truth_t_index[:,None,None,None],delta_obs_index[:,None,None],forget_index[:,None], EnsSize_index,:, metric]
            
            if nfilter>1:
                Z=Z[...,1]/Z[...,0]
                cmap=cmaps[0]
                vmax=np.exp(np.abs(np.log(Z)).max())
                vmin=1/vmax
                norm=colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                vmax=Z.max()
                vmin=Z.min()
                Z=Z[...,nfilter]
                cmap=cmaps[1]
                norm=colors.Normalize(vmin=vmin, vmax=vmax)
                
            #Z=np.log(Z[...,1]/Z[...,0])
            #vmax=Z.max()
            #vmin=Z.min()
            #vmax=np.exp(np.abs(np.log(Z)).max())
            #vmin=1/vmax
            #vmax=np.abs(Z).max()
            #vmin=-vmax
            #vmax=np.abs(np.log(Z)).max()
            #vmin=-vmax
            
            #norm=colors.FuncNorm((lambda x: np.log(x), lambda x: np.exp(x)),vmin=0.25, vmax=4)
            #norm=colors.LogNorm(vmin=vmin, vmax=vmax)
            fig, axs = plt.subplots(Z.shape[0], Z.shape[1], sharex=True, sharey=True, figsize=(12.8,19.4))
            #im=[[axs[i,j].imshow(Z[i,j], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower') for j in range(Z.shape[1])] for i in range(Z.shape[0])]
            im=[[axs[i,j].imshow(Z[i,j], cmap=cmap, norm=norm, origin='lower') for j in range(Z.shape[1])] for i in range(Z.shape[0])]
            
            for j, xtitle in enumerate(arrays['delta_obs_array'][delta_obs_index]):
                axs[0,j].set_title(f'{xtitle}')
                axs[-1,j].set_xlabel('EnsSize')
                axs[-1,j].set_xticks([float(x) for x in range(len(xticks))])
                axs[-1,j].set_xticklabels(xticks, rotation=45)
            for ax in axs[:, 0]:
                ax.set_ylabel('Forget')
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
            
            cbar=fig.colorbar(im[0][0], ax=axs)
            if nfilter==2:
                ticks=list(range(1,int(np.ceil(vmax))))
                ticks_list=[1/tick for tick in ticks[:0:-1]]+ticks
                ticks_labels=[f'1/{tick}' for tick in ticks[:0:-1]]+[str(tick) for tick in ticks]
                cbar.ax.set_yticks(ticks_list)
                cbar.ax.set_yticklabels(ticks_labels)
            
            fig.suptitle(title,fontsize=22)
            if save:
                fig.savefig(os.path.join(folder,title.replace(' ','_')))
    
    ######## overview #####################
    
    for step in range(1,3):
        Z=arrays['Z'][-1,delta_obs_index[:,None, None], forget_index[:,None], EnsSize_index,:, 1:]
        Z=np.concatenate((Z, Z[...,1:2,:]/Z[...,0:1,:]), axis=-2)
        
        Z=Z[...,::step,:]
        lab=['SEIK RMSE','GHOSH RMSE','RMSE ratio'][::step]
        
        Z=Z.transpose([-2,-1,0,1, 2]).reshape((-1,)+Z.shape[:3])
        
        fig, axs = plt.subplots(Z.shape[0], Z.shape[1], sharex=True, sharey=True, figsize=(12.8*3/4,19.4*3/4), dpi=300)
        
        cmap=cmaps[1]
        vmax=Z[:-2].max()
        vmin=Z[:-2].min()
        #norm=colors.Normalize(vmin=vmin, vmax=vmax)
        #vmax=np.exp(np.abs(np.log(Z[:-2])).max())
        #vmin=1/vmax
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
        im=[[axs[i,j].imshow(Z[i,j], cmap=cmap, norm=norm, origin='lower') for j in range(Z.shape[1])] for i in range(Z.shape[0]-2)]
        
        ticks_list=[]
        ticks_labels=[]
        for i in np.arange(np.floor(np.log10(vmin)),np.floor(np.log10(vmax))+1, dtype='int64'):
            ticks=np.arange(np.ceil(vmin*10**-i),10, dtype='int64')
            ticks=ticks[ticks<=vmax*10**-i]
            if i<0:
                ticks_labels+=['0.'+'0'*-(i+1)+str(tick) for tick in ticks]
                ticks=ticks*10.0**i
            else:
                ticks*=10**i
                ticks_labels+=[str(tick) for tick in ticks]
            ticks_list+=list(ticks)
        
        cmap=cmaps[0]
        vmax=np.exp(np.abs(np.log(Z[-2:])).max())
        vmin=1/vmax
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
        im.extend([[axs[i,j].imshow(Z[i,j], cmap=cmap, norm=norm, origin='lower') for j in range(Z.shape[1])] for i in range(Z.shape[0]-2, Z.shape[0])])
        
        cbars=[fig.colorbar(im[i][0], ax=axs[i:i+2], label=label) for i, label in zip(range(0,Z.shape[0],2),lab) ]
        for cbar, label in zip(cbars,lab):
            cbar.set_label(label, fontsize=18)
        
        #ticks=cbars[0].ax.get_yticks(minor=True)
        #ticks_labels=[str(tick) for tick in ticks_list]
        for i in range(len(cbars)-1):
            cbars[i].ax.set_yticks(ticks_list)
            cbars[i].ax.set_yticklabels(ticks_labels)
        
        ticks=list(range(1,int(np.ceil(vmax))))
        ticks_list=[1/tick for tick in ticks[:0:-1]]+ticks
        ticks_labels=[f'1/{tick}' for tick in ticks[:0:-1]]+[str(tick) for tick in ticks]
        cbars[-1].ax.set_yticks(ticks_list)
        cbars[-1].ax.set_yticklabels(ticks_labels)    
        
        for j, xtitle in enumerate(arrays['delta_obs_array'][delta_obs_index]):
            axs[0,j].set_title(f'{xtitle}')
            
            axs[-1,j].set_xlabel('EnsSize', size=14)
            axs[-1,j].set_xticks([float(x) for x in range(len(xticks))])
            axs[-1,j].set_xticklabels(xticks, rotation=45)
            
            #for i in range(len(cbars)):
                #axs[i*2+1,j].set_xlabel('EnsSize', size=14)
                #axs[i*2+1,j].set_xticks([float(x) for x in range(len(xticks))])
                #axs[i*2+1,j].set_xticklabels(xticks, rotation=45)
        
        for ax, lab in zip(axs[:, 0],['Assimilated', "Non-assimilated"]*(Z.shape[0]//2)):
            ax.set_ylabel('Forget', size=14)
            ax.set_yticks([float(x) for x in range(len(yticks))])
            ax.set_yticklabels(yticks)
            ax.annotate(lab, (0, 0.5), xytext=(-50, 0),
                        textcoords='offset points', xycoords='axes fraction',
                        ha='right', va='center', size=16, rotation=90) 
        for ax in axs.flatten():
            ax.tick_params(length=0.0)
            
        axs[0, len(delta_obs_index)//2].annotate('Obs Frequency', (0.5, 1), xytext=(0, 20),
                                textcoords='offset points', xycoords='axes fraction',
                                ha='center', va='bottom', size=16)
            
        title=f'Overview_{step}'
        #fig.suptitle(title,fontsize=22)
        if save:
            fig.savefig(os.path.join(folder,title.replace(' ','_')))
    
    ######## overview2 #####################
    if False:
        Z=arrays['Z'][-1,delta_obs_index[:,None],-1, EnsSize_index,:, 1:]
        Z=np.concatenate((Z, Z[...,1:2,:]/Z[...,0:1,:]), axis=-2)
        Z=Z.transpose([-2,-1,0,1])
        
        fig, axs = plt.subplots(Z.shape[0], Z.shape[1], sharex=True, sharey=True, figsize=(8,8))
        im=[[axs[i,j].imshow(Z[i,j], cmap=cmaps[1], vmax=Z[:-1].max(), vmin=Z[:-1].min(), origin='lower') for j in range(Z.shape[1])] for i in range(Z.shape[0]-1)]
        
        cmap=cmaps[0]
        vmax=np.exp(np.abs(np.log(Z[-1])).max())
        vmin=1/vmax
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
        im.append([axs[Z.shape[0]-1,j].imshow(Z[Z.shape[0]-1,j], cmap=cmap, norm=norm, origin='lower') for j in range(Z.shape[1])])
        
        cbars=[fig.colorbar(im[0][0], ax=axs[:Z.shape[0]-1]), fig.colorbar(im[Z.shape[0]-1][0], ax=axs[Z.shape[0]-1]) ]
        
        ticks=list(range(1,int(np.ceil(vmax))))
        ticks_list=[1/tick for tick in ticks[:0:-1]]+ticks
        ticks_labels=[f'1/{tick}' for tick in ticks[:0:-1]]+[str(tick) for tick in ticks]
        cbars[1].ax.set_yticks(ticks_list)
        cbars[1].ax.set_yticklabels(ticks_labels)    
        
        for j, xtitle in enumerate(['Observed', "Unobserved"]):
            axs[0,j].set_title(f'{xtitle}')
            axs[-1,j].set_xlabel('EnsSize')
            axs[-1,j].set_xticks([float(x) for x in range(len(xticks))])
            axs[-1,j].set_xticklabels(xticks, rotation=45)
        
        yticks=[str(x) for x in arrays['delta_obs_array'][delta_obs_index]]
        for ax, lab in zip(axs[:, 0], ['SEIK', 'GHOSH', 'Ratio']):
            ax.set_ylabel('Obs Frequency')
            ax.set_yticks([float(x) for x in range(len(yticks))])
            ax.set_yticklabels(yticks)
            ax.annotate(lab, (0, 0.5), xytext=(-40, 0),
                        textcoords='offset points', xycoords='axes fraction',
                        ha='right', va='center', size=14, rotation=90) 
        for ax in axs.flatten():
            ax.tick_params(length=0.0)
            
        title='Overview2'
        fig.suptitle(title,fontsize=22)
        if save:
            fig.savefig(os.path.join(folder,title.replace(' ','_')))
    
    plt.show()
    
if __name__=='__main__':
    print('start')
    N=2**6-2
    t_span=[0.0,20.0]
    obs_std=1.0
    n_experiments=100
    error_std=np.ones(N)*5.0
    model=DA.Lorenz96(F=8)

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
    #Z=list(np.load('Z.bkp.npy'))
    for truth_t in truth_t_array:
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
                    #Z.extend(list(metric_array[1]/metric_array[0]))
                    Z.extend(list(metric_array.flatten()))
                    
                    del ens_filters
                    del test
                    del metric_array
                    gc.collect()
        del IC_truth
        gc.collect()
        np.save('Z.bkp', Z)
        
    Z=np.reshape(Z, [len(truth_t_array),len(delta_obs_array),len(forget_array),len(EnsSize_array),2,3])
    print('saving Z')
    np.savez('Z.npz', Z=Z, truth_t_array=truth_t_array, delta_obs_array=delta_obs_array, forget_array=forget_array, EnsSize_array=EnsSize_array )
    print('Z.npz saved')
            
    #plot_results(Z)



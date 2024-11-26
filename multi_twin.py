
import numpy as np
import DA
import Models
import Metrics
from Filters import Seik, Ghosh

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

import gc
import os
import shutil
from warnings import warn

def main_long(model=Models.Lorenz96(), append_str='_96'):
    print('start')
    with np.load('Z'+append_str+'.npz') as f:
        arrays={}
        for key in f.files:
            arrays[key]=f[key]
    print('Z'+append_str+'.npz loaded')
            
    best=np.linalg.norm(arrays['Z'][...,0], axis=0, keepdims=False).argmin(axis=1, keepdims=False)
    best=arrays['forget_array'][best]
    
    N=2**6-2
    t_span=[0.0,150.0]
    obs_std=1.0
    n_experiments=100
    clim_error=True
    
    if clim_error:
        IC_0=model.climatological_moments(N)
    else:
        error_std=np.ones(N)*5.0
        IC_0=[0.01]+[0.0]*(N-1)
        IC_0,_ =model([0.0,20],IC_0)
        del _
        gc.collect()

    obs_each=2
    indices=range(0,N,obs_each)
    obs=DA.Observation(np.zeros(N//obs_each),np.ones(N//obs_each)*obs_std, indices=indices)
    #obs=DA.Observation(np.zeros([n_experiments,N//obs_each]),np.ones([n_experiments,N//obs_each])*obs_std, indices=indices)
    
    metrics=[
            DA.TimeMean(DA.RmpeByTime(index= None), name='RmseTot'),
            DA.TimeMean(DA.RmpeByTime(index= indices), name='RmseObserved'),
            DA.TimeMean(DA.RmpeByTime(index= tuple(set(range(N))-set(indices))), name='RmseNotObserved'),
            DA.TimeMean(Metrics.LikelihoodByTime(), name='Likelihood'),
            ]

    try:
        filename='Z_long'+append_str+'.npz'
        with np.load(filename) as f:
            saved={}
            for key in f.files:
                saved[key]=f[key]
    except FileNotFoundError as err:
        warn(f"There is no save file: {filename}. I will start computation from scratch.")
        saved={'Z':np.array([]), 
                'delta_obs_array':np.array([]),
                'EnsSize_array':np.array([], dtype=np.int64),
                'timing':np.array([]),
                }
    
    Z=[]
    timing=[]
    delta_obs_array=[0.1, 0.15, 0.2, 0.25, 0.3]
    EnsSize_array = list(2**np.arange(4,7)-1)
    
    delta_obs_array=list(arrays['delta_obs_array'])
    EnsSize_array = list(arrays['EnsSize_array'])
    
    #Z=list(np.load('Z'+append_str+'.bkp.npy'))
    #timing=list(np.load('timing'+append_str+'.bkp.npy'))
    
    print("Explored configurations:")
    print(f"delta_obs_array = {delta_obs_array}")
    print(f"EnsSize_array = {EnsSize_array}")
    i_conf=0
    n_conf=len(truth_t_array)*len(delta_obs_array)*len(forget_array)*len(EnsSize_array)
    for delta_obs in delta_obs_array:
        for EnsSize in EnsSize_array:
            i_conf+=1
            print(f"cofiguration {i_conf} of {n_conf}:")
            print(f'computing for delta_obs, EnsSize = {(delta_obs, EnsSize)}')
                    
            forget= best[np.nonzero(arrays['delta_obs_array']==delta_obs)[0][0],
                            np.nonzero(arrays['EnsSize_array']==EnsSize)[0][0]]
            
            if all([delta_obs in saved['delta_obs_array'], EnsSize in saved['EnsSize_array']]):
                Z.extend(list(saved['Z'][np.nonzero(saved['delta_obs_array']==delta_obs)[0][0],
                                            np.nonzero(saved['EnsSize_array']==EnsSize)[0][0]].flatten()))
                timing.extend(list(saved['timing'][np.nonzero(saved['delta_obs_array']==delta_obs)[0][0],
                                            np.nonzero(saved['EnsSize_array']==EnsSize)[0][0]].flatten()))
                print('read from saved file')
                continue
            
            ens_filters=[
                        Seik(EnsSize, forget=forget[0]),
                        Ghosh(EnsSize, order=5, forget=forget[1]),
                        ]
            test=DA.TwinExperiment(t_span, model, ens_filters, metrics=metrics)
            test.build_truth(IC_0, delta=delta_obs)
            test.build_obs(np.arange(t_span[0]+delta_obs,t_span[1],delta_obs), obs)
            test.build_tests()
            
            if clim_error:
                test.build_climatological_ICs(n_experiments=n_experiments)
            else:
                test.build_ICs(error_std, n_experiments=n_experiments)

            test.run()
            
            metric_array=np.array([[metric.result[0].item() for metric in filter_test.metrics_result] for filter_test in test.tests])
            #Z.extend(list(metric_array[1]/metric_array[0]))
            Z.extend(list(metric_array.flatten()))
            
            timing_array=np.array([[filter_test.counter.forecast, filter_test.counter.analysis, filter_test.counter.model, filter_test.counter.DA, filter_test.counter.total] 
                                    for filter_test in test.tests])
            timing.extend(list(timing_array.flatten()))
            
            del ens_filters
            del test
            del metric_array
            del timing_array
            gc.collect()
        np.save('Z'+append_str+'.bkp', Z)
        np.save('timing'+append_str+'.bkp', timing)
        
    Z=np.reshape(Z, [len(delta_obs_array), len(EnsSize_array), 2, len(metrics)])
    timing=np.reshape(timing, [len(delta_obs_array), len(EnsSize_array), 2, 5])
    print('saving Z_long'+append_str)
    np.savez('Z_long'+append_str+'.npz', Z=Z, delta_obs_array=delta_obs_array, EnsSize_array=EnsSize_array, timing=timing, best=best)
    print('Z_long'+append_str+'.npz saved')
                
def plot_results_long(arrays='Z_long.npz', delta_obs_array=None, EnsSize_array=None, titles=['Total RMSE','Assimilated RMSE','Non-assimilated RMSE'], cmaps=['seismic_r','viridis'], max_metric=4.0, save=False, folder='SAVED'):
    
    if save:
        if os.path.exists(folder):
            max_answers=3
            for _ in range(max_answers):
                yn=input(f'The folder {folder} already exists and it will be overwritten. Are you sure (y/n)? ')
                if yn=='n':
                    print('Aborted!')
                    return
                elif yn=='y':
                    break
                else:
                    print('Unrecognized answer, please write "y" or "n".')
            else:
                print('Maximum number of wrong answers. Aborted!')
                return
        else:
            os.makedirs(folder)
            if type(arrays)==type('hello'):
                shutil.copy2(arrays,folder)
    
    if type(arrays)==type('hello'):
        with np.load(arrays) as f:
            arrays={}
            for key in f.files:
                arrays[key]=f[key]
                
    if delta_obs_array is None:
        delta_obs_index=np.arange(len(arrays['delta_obs_array']))
    else:
        delta_obs_index=np.array([i for i, val in enumerate(arrays['delta_obs_array']) if val in delta_obs_array])
    if EnsSize_array is None:
        EnsSize_index=np.arange(len(arrays['EnsSize_array']))
    else:
        EnsSize_index=np.array([i for i, val in enumerate(arrays['EnsSize_array']) if val in EnsSize_array])       
    
    xtitles=arrays['delta_obs_array'][delta_obs_index]
    xticks=[str(x) for x in arrays['EnsSize_array'][EnsSize_index]]
    yticks=['Assimilated', 'Non-assimilated']#, 'Likelihood']
    
    ######## overview #####################
    
    Z=arrays['Z'][delta_obs_index[:,None], EnsSize_index,:, 1:3]
    Z=np.concatenate((Z, Z[...,1:2,:]/Z[...,0:1,:]), axis=-2)
    
    lab=['SEIK RMSE','GHOSH RMSE','RMSE ratio']
    
    Z=Z.transpose([-2,0,-1, 1])
    
    fig, axs = plt.subplots(Z.shape[0], Z.shape[1], sharex=False, sharey=True, squeeze=False, figsize=(12.8*3/4,19.4*3/4), dpi=300)
    
    cmap=cmaps[1]
    vmax=min([Z[:-1].max(), max_metric])
    vmin=Z[:-1].min()
    #norm=colors.Normalize(vmin=vmin, vmax=vmax)
    #vmax=np.exp(np.abs(np.log(Z[:-2])).max())
    #vmin=1/vmax
    
    log_color=False
    if log_color:
        norm=colors.LogNorm(vmin=vmin, vmax=vmax)
        ticks_list=[]
        ticks_labels=[]
        for i in np.arange(np.floor(np.log10(vmin)),np.floor(np.log10(vmax))+1, dtype='int64'):
            ticks=np.arange(np.ceil(vmin*10.0**-i),10, dtype='int64')
            ticks=ticks[ticks<=vmax*10.0**-i]
            if i<0:
                ticks_labels+=['0.'+'0'*-(i+1)+str(tick) for tick in ticks]
                ticks=ticks*10.0**i
            else:
                ticks*=10**i
                ticks_labels+=[str(tick) for tick in ticks]
            ticks_list+=list(ticks)
    else:
        norm=colors.Normalize(vmin=vmin, vmax=vmax)
        k=10**np.floor(np.log10(vmax-vmin)).astype('int64')
        ticks_list=range(np.ceil(vmin/k).astype('int64'),np.floor(vmax/k).astype('int64')+1 )
        ticks_list=[tick*k for tick in ticks_list]
        ticks_labels=[str(tick) for tick in ticks_list]
    
    im=[[axs[i,j].imshow(Z[i,j], cmap=cmap, norm=norm, origin='upper') for j in range(Z.shape[1])] for i in range(Z.shape[0]-1)]
    
    cmap=cmaps[0]
    vmax=max([np.exp(np.abs(np.log(Z[-1])).max()), 2])
    vmin=1/vmax
    norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    im.extend([[axs[i,j].imshow(Z[i,j], cmap=cmap, norm=norm, origin='upper') for j in range(Z.shape[1])] for i in range(Z.shape[0]-1, Z.shape[0])])
    
    cbars=[fig.colorbar(im[i][0], ax=axs[i:i+1], label=label, location='bottom') for i, label in zip(range(0,Z.shape[0],1),lab) ]
    for cbar, label in zip(cbars,lab):
        cbar.set_label(label, fontsize=18)
    
    #ticks=cbars[0].ax.get_yticks(minor=True)
    #ticks_labels=[str(tick) for tick in ticks_list]
    for i in range(len(cbars)-1):
        cbars[i].ax.set_xticks(ticks_list)
        cbars[i].ax.set_xticklabels(ticks_labels)
    
    #ticks=list(range(1,int(np.ceil(vmax))))
    #ticks_list=[1/tick for tick in ticks[:0:-1]]+ticks
    #ticks_labels=[f'1/{tick}' for tick in ticks[:0:-1]]+[str(tick) for tick in ticks]
    
    ticks_list=[]
    ticks_labels=[]
    for i in np.arange(np.floor(np.log10(vmin)),np.floor(np.log10(vmax))+1, dtype='int64'):
        ticks=np.arange(np.ceil(vmin*10.0**-i),10, dtype='int64')
        ticks=ticks[ticks<=vmax*10.0**-i]
        if i<0:
            ticks_labels+=['0.'+'0'*-(i+1)+str(tick) for tick in ticks]
            ticks=ticks*10.0**i
        else:
            ticks*=10**i
            ticks_labels+=[str(tick) for tick in ticks]
        ticks_list+=list(ticks)
        
    cbars[-1].ax.set_xticks(ticks_list)
    cbars[-1].ax.set_xticklabels(ticks_labels)
    
    for j, xtitle in enumerate(arrays['delta_obs_array'][delta_obs_index]):
        axs[0,j].set_title(f'{xtitle}')
        
        for i in range(Z.shape[0]):
            axs[i,j].set_xlabel('EnsSize', size=14)
            axs[i,j].set_xticks([float(x) for x in range(len(xticks))])
            axs[i,j].set_xticklabels(xticks, rotation=45)
        
    for ax in axs[:,0]:
        ax.set_yticks([float(x) for x in range(len(yticks))])
        ax.set_yticklabels(yticks)
        
    for ax in axs.flatten():
        ax.tick_params(length=0.0)
        
    axs[0, len(delta_obs_index)//2].annotate('Obs Frequency', (0.5, 1), xytext=(0, 20),
                            textcoords='offset points', xycoords='axes fraction',
                            ha='center', va='bottom', size=16)
        
    title=f'Overview'
    #fig.suptitle(title,fontsize=22)
    #fig.set_tight_layout(True)
    if save:
        fig.savefig(os.path.join(folder,title.replace(' ','_')))
    
    plt.show()

def plot_results(arrays=None, truth_t_array=None, delta_obs_array=None, forget_array=None, EnsSize_array=None, titles=['Total RMSE','Assimilated RMSE','Non-assimilated RMSE'], cmaps=['seismic_r','viridis'], max_metric=4.0, save=False, folder='SAVED'):
    
    if save:
        if os.path.exists(folder):
            max_answers=3
            for _ in range(max_answers):
                yn=input(f'The folder {folder} already exists and it will be overwritten. Are you sure (y/n)? ')
                if yn=='n':
                    print('Aborted!')
                    return
                elif yn=='y':
                    break
                else:
                    print('Unrecognized answer, please write "y" or "n".')
            else:
                print('Maximum number of wrong answers. Aborted!')
                return
        else:
            os.makedirs(folder)
            if arrays is None:
                shutil.copy2('Z.npz',folder)
    
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
                                np.take_along_axis(arrays['Z'], np.linalg.norm(arrays['Z'][...,0:1], axis=0, keepdims=True).argmin(axis=2, keepdims=True), axis=2)),
                                axis=2)
    forget_index=np.concatenate((forget_index,[-1]))
    
    arrays['Z']=np.concatenate((arrays['Z'],np.sqrt((arrays['Z']**2).mean(axis=0, keepdims=True))),axis=0)
    #truth_t_index=np.concatenate((truth_t_index,[-1]))
    
    if False:
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
                    vmax=min([Z.max(), max_metric])
                    vmin=Z.min()
                    Z=Z[...,nfilter]
                    cmap=cmaps[1]
                    #norm=colors.Normalize(vmin=vmin, vmax=vmax)
                    norm=colors.LogNorm(vmin=vmin, vmax=vmax)
                    
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
                fig, axs = plt.subplots(Z.shape[0], Z.shape[1], sharex=True, sharey=True, squeeze=False, figsize=(12.8,19.4))
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
        #Z=arrays['Z'][-1,delta_obs_index[:,None, None], forget_index[:,None], EnsSize_index,:, 1:]
        Z=np.sqrt(np.mean(arrays['Z'][:,delta_obs_index[:,None, None], forget_index[:,None], EnsSize_index,:, 1:3]**2, axis=0))
        Z=np.concatenate((Z, Z[...,1:2,:]/Z[...,0:1,:]), axis=-2)
        
        Z=Z[...,::step,:]
        lab=['SEIK RMSE','GHOSH RMSE','RMSE ratio'][::step]
        
        Z=Z.transpose([-2,-1,0,1, 2]).reshape((-1,)+Z.shape[:3])
        
        fig, axs = plt.subplots(Z.shape[0], Z.shape[1], sharex=True, sharey=True, squeeze=False, figsize=(12.8*3/4,19.4*3/4), dpi=300)
        
        cmap=cmaps[1]
        vmax=min([Z[:-2].max(), max_metric])
        vmin=Z[:-2].min()
        #norm=colors.Normalize(vmin=vmin, vmax=vmax)
        #vmax=np.exp(np.abs(np.log(Z[:-2])).max())
        #vmin=1/vmax
        
        log_color=False
        if log_color:
            norm=colors.LogNorm(vmin=vmin, vmax=vmax)
            ticks_list=[]
            ticks_labels=[]
            for i in np.arange(np.floor(np.log10(vmin)),np.floor(np.log10(vmax))+1, dtype='int64'):
                ticks=np.arange(np.ceil(vmin*10.0**-i),10, dtype='int64')
                ticks=ticks[ticks<=vmax*10.0**-i]
                if i<0:
                    ticks_labels+=['0.'+'0'*-(i+1)+str(tick) for tick in ticks]
                    ticks=ticks*10.0**i
                else:
                    ticks*=10**i
                    ticks_labels+=[str(tick) for tick in ticks]
                ticks_list+=list(ticks)
        else:
            norm=colors.Normalize(vmin=vmin, vmax=vmax)
            k=10**np.floor(np.log10(vmax-vmin)).astype('int64')
            ticks_list=range(np.ceil(vmin/k).astype('int64'),np.floor(vmax/k).astype('int64')+1 )
            ticks_list=[tick*k for tick in ticks_list]
            ticks_labels=[str(tick) for tick in ticks_list]
        
        im=[[axs[i,j].imshow(Z[i,j], cmap=cmap, norm=norm, origin='lower') for j in range(Z.shape[1])] for i in range(Z.shape[0]-2)]
        
        
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
            
        if False:
            ticks=list(range(1,int(np.ceil(vmax))))
            ticks_list=[1/tick for tick in ticks[:0:-1]]+ticks
            ticks_labels=[f'1/{tick}' for tick in ticks[:0:-1]]+[str(tick) for tick in ticks]
        else:
            ticks_list=[]
            ticks_labels=[]
            for i in np.arange(np.floor(np.log10(vmin)),np.floor(np.log10(vmax))+1, dtype='int64'):
                ticks=np.arange(np.ceil(vmin*10.0**-i),10, dtype='int64')
                ticks=ticks[ticks<=vmax*10.0**-i]
                if i<0:
                    ticks_labels+=['0.'+'0'*-(i+1)+str(tick) for tick in ticks]
                    ticks=ticks*10.0**i
                else:
                    ticks*=10**i
                    ticks_labels+=[str(tick) for tick in ticks]
                ticks_list+=list(ticks)
                
        
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
    
def main(model=Models.Lorenz96()):
    print('start')
    N=2**6-2
    t_span=[0.0,20.0]
    obs_std=1.0
    n_experiments=100
    clim_error=True
    
    if clim_error:
        IC_0=model.climatological_moments(N)
    else:
        error_std=np.ones(N)*5.0
        IC_0=[0.01]+[0.0]*(N-1)

    obs_each=2
    indices=range(0,N,obs_each)
    obs=DA.Observation(np.zeros(N//obs_each),np.ones(N//obs_each)*obs_std, indices=indices)
    #obs=DA.Observation(np.zeros([n_experiments,N//obs_each]),np.ones([n_experiments,N//obs_each])*obs_std, indices=indices)
    
    metrics=[
            DA.HalfTimeMean(DA.RmpeByTime(index= None), name='RmseTot'),
            DA.HalfTimeMean(DA.RmpeByTime(index= indices), name='RmseObserved'),
            DA.HalfTimeMean(DA.RmpeByTime(index= tuple(set(range(N))-set(indices))), name='RmseNotObserved'),
            DA.TimeMean(Metrics.LikelihoodByTime(), name='Likelihood'),
            ]

    #truth_t_array, delta_obs_array, forget_array, EnsSize_array = np.meshgrid([20.0,40.0,60.0], np.linspace(0.1, 0.3, 3), np.linspace(0.7, 1.0, 4), 2**np.arange(3,7)-1, indexing='ij')
    #for truth_t, delta_obs, forget, EnsSize in zip(truth_t_array.flatten(), delta_obs_array.flatten(), forget_array.flatten(), EnsSize_array.flatten()):
    
    try:
        filename='Z.npz'
        with np.load(filename) as f:
            saved={}
            for key in f.files:
                saved[key]=f[key]
    except FileNotFoundError as err:
        warn(f"There is no save file: {filename}. I will start computation from scratch.")
        saved={'Z':np.array([]), 
                'truth_t_array':np.array([]),
                'delta_obs_array':np.array([]),
                'forget_array':np.array([]),
                'EnsSize_array':np.array([], dtype=np.int64),
                'timing':np.array([]),
                }
    
    Z=[]
    timing=[]
    truth_t_array=[20.0, 40.0, 60.0, 80.0]
    delta_obs_array=[0.1, 0.15, 0.2, 0.25, 0.3]
    forget_array=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    #forget_array=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    #forget_array=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    EnsSize_array = list(2**np.arange(4,7)-1)
    
    #Z=list(np.load('Z.bkp.npy'))
    #timing=list(np.load('timing.bkp.npy'))
    
    print("Explored configurations:")
    print(f"truth_t_array = {truth_t_array}")
    print(f"delta_obs_array = {delta_obs_array}")
    print(f"forget_array = {forget_array}")
    print(f"EnsSize_array = {EnsSize_array}")
    i_conf=0
    n_conf=len(truth_t_array)*len(delta_obs_array)*len(forget_array)*len(EnsSize_array)
    for truth_t in truth_t_array:
        IC_truth,_ =model([0.0,truth_t],IC_0)
        del _
        gc.collect()
        for delta_obs in delta_obs_array:
            for forget in forget_array:
                for EnsSize in EnsSize_array:
                    i_conf+=1
                    print(f"cofiguration {i_conf} of {n_conf}:")
                    print(f'computing for truth_t, delta_obs, forget, EnsSize = {(truth_t, delta_obs, forget, EnsSize)}')
                    
                    if all([truth_t in saved['truth_t_array'], delta_obs in saved['delta_obs_array'], forget in saved['forget_array'], EnsSize in saved['EnsSize_array']]):
                        Z.extend(list(saved['Z'][np.nonzero(saved['truth_t_array']==truth_t)[0][0],
                                                 np.nonzero(saved['delta_obs_array']==delta_obs)[0][0],
                                                 np.nonzero(saved['forget_array']==forget)[0][0],
                                                 np.nonzero(saved['EnsSize_array']==EnsSize)[0][0]].flatten()))
                        timing.extend(list(saved['timing'][np.nonzero(saved['truth_t_array']==truth_t)[0][0],
                                                    np.nonzero(saved['delta_obs_array']==delta_obs)[0][0],
                                                    np.nonzero(saved['forget_array']==forget)[0][0],
                                                    np.nonzero(saved['EnsSize_array']==EnsSize)[0][0]].flatten()))
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
                    
                    if clim_error:
                        test.build_climatological_ICs(n_experiments=n_experiments)
                    else:
                        test.build_ICs(error_std, n_experiments=n_experiments)

                    test.run()
                    
                    metric_array=np.array([[metric.result[0].item() for metric in filter_test.metrics_result] for filter_test in test.tests])
                    #Z.extend(list(metric_array[1]/metric_array[0]))
                    Z.extend(list(metric_array.flatten()))
                    
                    timing_array=np.array([[filter_test.counter.forecast, filter_test.counter.analysis, filter_test.counter.model, filter_test.counter.DA, filter_test.counter.total] 
                                           for filter_test in test.tests])
                    timing.extend(list(timing_array.flatten()))
                    
                    del ens_filters
                    del test
                    del metric_array
                    del timing_array
                    gc.collect()
        del IC_truth
        gc.collect()
        np.save('Z.bkp', Z)
        np.save('timing.bkp', timing)
        
    Z=np.reshape(Z, [len(truth_t_array), len(delta_obs_array), len(forget_array), len(EnsSize_array), 2, len(metrics)])
    timing=np.reshape(timing, [len(truth_t_array), len(delta_obs_array), len(forget_array), len(EnsSize_array), 2, 5])
    print('saving Z')
    np.savez('Z.npz', Z=Z, truth_t_array=truth_t_array, delta_obs_array=delta_obs_array, forget_array=forget_array, EnsSize_array=EnsSize_array, timing=timing)
    print('Z.npz saved')
            
    #plot_results(Z)
    

if __name__=='__main__':
    main()
    plot_results()
    
    main_long()
    plot_results_long()
    
    #main_long(model=Models.Lorenz05(two_scale=True), append_str='_2005')
    



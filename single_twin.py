import numpy as np

import DA
from Filters import Seik, Ghosh
import Metrics
import Models

import gc
    
def main(N=62, EnsSize=31, delta_obs=0.15, forget=0.75, n_experiments=100, model=Models.Lorenz96(), clim_error=True):
    obs_each=2
    indices=range(0,N,obs_each)
    obs_std=1.0
    t_span=[0.0,20.0]
    true_std_sigma=0
    
    if clim_error:
        IC_0=model.climatological_moments(N)
    else:
        error_std=np.ones(N)*5.0
        IC_0=[0.01]+[0.0]*(N-1)
        IC_0,_ =model([0.0,20],IC_0)
        del _
        gc.collect()
    
    IC_truth,_ =model([0.0,20],IC_0)

    #obs=DA.ObsByIndex(np.zeros(N//obs_each),np.ones(N//obs_each)*obs_std, indices=indices)
    obs=DA.ObsByIndex(np.zeros([n_experiments, len(indices)]),np.ones([n_experiments, len(indices)])*obs_std, true_std=np.ones([n_experiments, 1])*obs_std, indices=indices)

    ens_filters=[
                #Seik(EnsSize, forget=1.0),
                #Seik(EnsSize, forget=0.95),
                #Seik(EnsSize, forget=0.9), 
                #Seik(EnsSize, forget=0.85),
                #Seik(EnsSize, forget=0.8), 
                #Seik(EnsSize, forget=0.75),
                #Seik(EnsSize, forget=0.7),
                Seik(EnsSize, forget=forget),
                #Ghosh(EnsSize, order=5, forget=1.0),
                #Ghosh(EnsSize, order=5, forget=0.95),
                #Ghosh(EnsSize, order=5, forget=0.9), 
                #Ghosh(EnsSize, order=5, forget=0.85),
                #Ghosh(EnsSize, order=5, forget=0.8),
                #Ghosh(EnsSize, order=5, forget=0.7),
                Ghosh(EnsSize, order=5, forget=forget),
                #Seik(EnsSize, with_autotuning=True),
                #Ghosh(EnsSize, order=5, with_autotuning=True),
                ]

    metrics=[
            DA.RmpeByTime(index= None, name='RMSE all'),
            DA.RmpeByTime(index= indices, name='RMSE assimilated'),
            DA.RmpeByTime(index= tuple(set(range(N))-set(indices)), name='RMSE non-assimilated'),
            #Metrics.LikelihoodByTime(name='Log-Likelihood'),
            #Metrics.Cumulative(Metrics.LikelihoodByTime(), name='Cumulative Log-Likelihood'),
            DA.HalfTimeMean(DA.RmpeByTime(index= None), name='RMSE all'),
            DA.HalfTimeMean(DA.RmpeByTime(index= indices), name='RMSE assimilated'),
            DA.HalfTimeMean(DA.RmpeByTime(index= tuple(set(range(N))-set(indices))), name='RMSE non-assimilated'),
            #DA.TimeMean(Metrics.LikelihoodByTime(), name='Log-Likelihood'),
            ]


    test=DA.TwinExperiment(t_span, model, ens_filters, metrics=metrics)

    test.build_truth(IC_truth, delta=delta_obs)
    test.build_obs(np.arange(t_span[0]+delta_obs,t_span[1],delta_obs), obs, true_std_sigma=true_std_sigma)
    test.build_tests()
    
    if clim_error:
        test.build_climatological_ICs(n_experiments=n_experiments)
    else:
        test.build_ICs(error_std, n_experiments=n_experiments)

    test.run()

    test.table()
    
    return test

if __name__=='__main__':
    test=main(forget=0.6, n_experiments=100)
    #test=main(model=Models.Lorenz05(two_scale=True))
    
    test.plot(ivar=2, draw_std=False, draw_metrics=False, show=False)
    test.plot(ivar=1, draw_std=False, draw_metrics=False, show=False)
    test.plot(draw_std=False, draw_var=False, show=True)
    #test.plot(ivar=np.s_[:], draw_std=True, draw_var=False)
     

import numpy as np

import DA
from Filters import Seik, Ghosh, EnsFilter
import Metrics
import Models2

import gc

# import matplotlib.pyplot as plt

class CtrlRun(EnsFilter):
    def __init__(self, EnsSize, weights=None, forget=1.0, with_autotuning=False, autotuning_bounds=None):
        super().__init__(EnsSize, weights, forget, with_autotuning, autotuning_bounds)
        self.seik=Seik(EnsSize)
    
    def sampling(self, mean_and_base):
        return self.seik.sampling(mean_and_base)
    
def main(EnsSize=5, delta_obs=1.0, forget=1.0, n_experiments=1, model=Models2.LogLotkaVolterra(N=2), clim_error=False):
    N=model.N
    indices=(1,)
    obs_std=0.5
    t_span=[0.0,40.0]
    true_std_sigma=0
    
    if clim_error:
        IC_0=model.climatological_moments(N)
    else:
        IC_truth=np.array([[2.0,2],
                           [1,-1],
                           [0,1],
                           [-1,0]])
        IC_truth[0]=np.log(IC_truth[0])
        IC_truth=IC_truth.flatten()
        error_std=np.array([[1.0,1],
                           [1,1],
                           [0,0],
                           [0,0]]).flatten()*0.5
        

    #obs=DA.Observation(np.zeros(N//obs_each),np.ones(N//obs_each)*obs_std, indices=indices)
    obs=DA.Observation(np.zeros([n_experiments, len(indices)]),np.ones([n_experiments, len(indices)])*obs_std, true_std=np.ones([n_experiments, 1])*obs_std, indices=indices)

    ens_filters=[
                #Seik(EnsSize, forget=1.0),
                #Seik(EnsSize, forget=0.95),
                #Seik(EnsSize, forget=0.9), 
                #Seik(EnsSize, forget=0.85),
                #Seik(EnsSize, forget=0.8), 
                #Seik(EnsSize, forget=0.75),
                #Seik(EnsSize, forget=0.7),
                # Seik(EnsSize, forget=forget),
                #Ghosh(EnsSize, order=5, forget=1.0),
                #Ghosh(EnsSize, order=5, forget=0.95),
                #Ghosh(EnsSize, order=5, forget=0.9), 
                #Ghosh(EnsSize, order=5, forget=0.85),
                #Ghosh(EnsSize, order=5, forget=0.8),
                #Ghosh(EnsSize, order=5, forget=0.7),
                Ghosh(EnsSize, order=5, forget=forget),
                #Seik(EnsSize, with_autotuning=True),
                #Ghosh(EnsSize, order=5, with_autotuning=True),
                CtrlRun(EnsSize),
                ]

    metrics=[
            DA.RmpeByTime(index= (0,1), name='RMSE state'),
            DA.RmpeByTime(index= tuple(range(2,8)), name='RMSE params'),
            #Metrics.LikelihoodByTime(name='Log-Likelihood'),
            #Metrics.Cumulative(Metrics.LikelihoodByTime(), name='Cumulative Log-Likelihood'),
            DA.HalfTimeMean(DA.RmpeByTime(index= None), name='RMSE all'),
            DA.HalfTimeMean(DA.RmpeByTime(index= indices), name='RMSE assimilated'),
            DA.HalfTimeMean(DA.RmpeByTime(index= tuple(set(range(N))-set(indices))), name='RMSE non-assimilated'),
            #DA.TimeMean(Metrics.LikelihoodByTime(), name='Log-Likelihood'),
            ]


    test=DA.TwinExperiment(t_span, model, ens_filters, metrics=metrics)

    ref=test.build_truth(IC_truth, delta=delta_obs)
    # print(ref.y[...,:10])
    # fig, ax=plt.subplots()
    # ax.plot(ref.t, ref.y[0])
    # ax.plot(ref.t, ref.y[1])
    # t=np.linspace(ref.t[0],ref.t[-1], 1000)
    # y=ref.sol(t)
    # ax.plot(t, y[0])
    # ax.plot(t, y[1])
    # plt.show()
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
    test=main(forget=0.95, n_experiments=1)
    #test=main(model=Models.Lorenz05(two_scale=True))
    # print(test.reference.y[...,0])
    for ivar in np.where(test.reference.y[...,0]!=0)[0]:
        test.plot(ivar=ivar, draw_std=False, draw_metrics=False, show=False)
    test.plot(draw_std=False, draw_var=False, show=True)
    #test.plot(ivar=np.s_[:], draw_std=True, draw_var=False)
     


import numpy as np
import DA
from Filters import Seik, Ghosh
from Metrics import LikelihoodByTime
    
EnsSize=31 #31
N=62 #30
nvars=31 #10
obs_each=2
indices=range(0,N,obs_each)
obs_std=1.0
delta_obs=0.15 #0.2
t_span=[0.0,20.0]
forget=0.7
atol=None
#atol=1.0e-6
n_experiments=100 #100

#error_std=np.flip(np.exp(-np.arange(N)))*0.5
#error_std=np.exp(-np.arange(N)*1.0)*1.0
error_std=np.ones(N)*5.0
#error_std[error_std<1.0e-3]=1.0e-3

Q_std=error_std*0.02
Q_std[Q_std<1.0e-4]=1.0e-4
Q_std=None

obs=DA.ObsByIndex(np.zeros(N//obs_each),np.ones(N//obs_each)*obs_std, indices=indices)

ens_filters=[
             #Seik(EnsSize, forget=1.0),
             #Seik(EnsSize, forget=0.95),
             ##Seik(EnsSize, forget=0.9), #best 0.05
             #Seik(EnsSize, forget=0.85),
             #Seik(EnsSize, forget=0.8), #best 0.1
             #Seik(EnsSize, forget=0.75),
             #Seik(EnsSize, forget=0.7),
             Seik(EnsSize, forget=forget),
             #Ghosh(EnsSize, order=5, forget=1.0),
             #Ghosh(EnsSize, order=5, forget=0.95),
             ##Ghosh(EnsSize, order=5, forget=0.9), #best 0.05-0.1
             #Ghosh(EnsSize, order=5, forget=0.85),
             #Ghosh(EnsSize, order=5, forget=0.8),
             #Ghosh(EnsSize, order=5, forget=0.7),
             Ghosh(EnsSize, order=5, forget=forget),
             ]

metrics=[DA.RmpeByTime(index= None, name='RmseTotByTime'),
         DA.RmpeByTime(index= indices, name='RmseObseervedByTime'),
         DA.RmpeByTime(index= tuple(set(range(N))-set(indices)), name='RmseNotObservedByTime'),
         LikelihoodByTime(name='LikelihoodByTime'),
        DA.HalfTimeMean(DA.RmpeByTime(index= None), name='RmseTot'),
        DA.HalfTimeMean(DA.RmpeByTime(index= indices), name='RmseObserved'),
        DA.HalfTimeMean(DA.RmpeByTime(index= tuple(set(range(N))-set(indices))), name='RmseNotObserved'),
        DA.TimeMean(LikelihoodByTime(), name='Likelihood'),
        ]

model=DA.Lorenz96() #, atol=atol)

Q_std_t=lambda t:None if t==0 else Q_std
test=DA.TwinExperiment(t_span, model, ens_filters, Q_std_t=Q_std_t, metrics=metrics)

IC_truth,_ =model([0.0,60.0],[0.01]+[0.0]*(N-1))
test.build_truth(IC_truth, delta=delta_obs, Q_std=Q_std)
test.build_obs(np.arange(t_span[0]+delta_obs,t_span[1],delta_obs), obs)
test.build_tests()
test.build_ICs(error_std, n_experiments=n_experiments)

test.run()

test.table()

test.plot(ivar=2, iexp=0, draw_std=False, draw_metrics=False, show=False)
test.plot(ivar=1, iexp=0, draw_std=False, draw_metrics=False, show=False)
test.plot(draw_std=False, draw_var=False)
#test.plot(ivar=15)


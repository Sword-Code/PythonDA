
import numpy as np
import DA
from Filters import Seik, Ghosh
from Metrics import LikelihoodByTime
import gc
import datetime
    
EnsSize=31 #31
N=40 #62 #30
nvars=31 #10
obs_each=2
indices=range(0,N,obs_each)
obs_std=1.0
delta_obs=0.1 #0.15 #0.2
t_span=[0.0,25.0]
forget=0.85 #0.7
atol=None
#atol=1.0e-6
n_experiments=100

#error_std=np.flip(np.exp(-np.arange(N)))*0.5
#error_std=np.exp(-np.arange(N)*1.0)*1.0
error_std=np.ones(N)*5.0
#error_std[error_std<1.0e-3]=1.0e-3

Q_std=error_std*0.02
Q_std[Q_std<1.0e-4]=1.0e-4
Q_std=None

#obs=DA.ObsByIndex(np.zeros([n_experiments,N//obs_each]),np.ones([n_experiments,N//obs_each])*obs_std, indices=indices)
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
             #Ghosh(EnsSize, order=5, forget=forget),
             ]

metrics=[DA.RmpeByTime(index= None, name='RmseTotByTime'),
         #DA.RmpeByTime(index= indices, name='RmseObseervedByTime'),
         #DA.RmpeByTime(index= tuple(set(range(N))-set(indices)), name='RmseNotObservedByTime'),
         #LikelihoodByTime(name='LikelihoodByTime'),
        DA.HalfTimeMean(DA.RmpeByTime(index= None), name='RmseTot'),
        #DA.HalfTimeMean(DA.RmpeByTime(index= indices), name='RmseObserved'),
        #DA.HalfTimeMean(DA.RmpeByTime(index= tuple(set(range(N))-set(indices))), name='RmseNotObserved'),
        #DA.TimeMean(LikelihoodByTime(), name='Likelihood'),
        ]

model=DA.Lorenz96() #, atol=atol)

Q_std_t=lambda t:None if t==0 else Q_std
test=DA.TwinExperiment(t_span, model, ens_filters, Q_std_t=Q_std_t, metrics=metrics)

IC_truth,_ =model([0.0,20.0],[0.01]+[0.0]*(N-1))
IC_truth,history=model([0.0,1000.0],IC_truth)
samples=history.sol(np.arange(0.0,1000.0,0.1))
cov=np.cov(samples)
print("cov diagonal:")
print(np.diagonal(cov))
eigenvalues, eigenvectors = np.linalg.eigh(cov)
eigenvalues=eigenvalues[::-1]
print("Eigenvalues:")
print(eigenvalues)
eigenvectors=eigenvectors[:,::-1]

#IC_truth,_ =model([0.0,60.0],[0.01]+[0.0]*(N-1))
test.build_truth(IC_truth, delta=delta_obs, Q_std=Q_std)
test.build_obs(np.arange(t_span[0]+delta_obs,t_span[1],delta_obs), obs)
test.build_tests()

test.build_ICs(error_std, n_experiments=n_experiments)

test.run()

test.table()

#test.plot(ivar=2, iexp=0, draw_std=False, draw_metrics=False, show=False)
#test.plot(ivar=1, iexp=0, draw_std=False, draw_metrics=False, show=False)
#test.plot(draw_std=False, draw_var=False)
#test.plot(ivar=15)
test.plot(ivar=np.s_[:], draw_var=False, draw_std=True, show=False, title='Diagonal const')

test.build_tests()
gc.collect()
error_std=np.sqrt(eigenvalues)
test.build_ICs(error_std, n_experiments=n_experiments)

test.run()

test.table()

#test.plot(ivar=2, iexp=0, draw_std=False, draw_metrics=False, show=False)
#test.plot(ivar=1, iexp=0, draw_std=False, draw_metrics=False, show=False)
#test.plot(draw_std=False, draw_var=False)
#test.plot(ivar=15)
test.plot(ivar=np.s_[:], draw_var=False, draw_std=True, show=False, title='Diagonal non-const')

def build_IC(self, eigenvalues, eigenvectors, mean=None):
    if mean is None:
        mean=self.reference.y[...,0]
    #error_std=mean.copy()
    #error_std[...]=std
    
    #print('error_std:')
    #print(error_std[:,:2])
    
    mean_and_base=np.zeros(mean.shape[:-1]+(self.ens_filter.EnsSize,mean.shape[-1]))
    mean_and_base[...,0,:]=mean
    mean_and_base[...,1:,:]=(eigenvectors[:,:self.ens_filter.EnsSize-1] *np.sqrt(eigenvalues[:self.ens_filter.EnsSize-1])).transpose()
    
    #mean_and_base[...,np.arange(1,self.ens_filter.EnsSize),np.arange(self.ens_filter.EnsSize-1)]=error_std*np.sqrt(self.ens_filter.forget)
    
    #indices=np.flip(np.argsort(error_std), axis=-1)[...,:self.ens_filter.EnsSize-1]
    #adv_slices=np.zeros((indices.ndim,)+ indices.shape, dtype=int)
    #for i, temp in enumerate(adv_slices):
        #temp[...]=np.arange(indices.shape[i])[np.index_exp[...]+(None,)*(indices.ndim-1-i)]
    #temp=mean_and_base[...,1:,:]
    #temp[np.index_exp[...]+(*adv_slices, indices)]=np.take_along_axis(error_std, indices, axis=-1)
    
    #print('mean_and_base:')
    #print(mean_and_base[0,:,:2])
    #print(np.matmul(utils.transpose(mean_and_base[...,1:,:]),mean_and_base[...,1:,:]))
    
    self.IC=self.ens_filter.sampling(mean_and_base)
    
    #print('IC:')
    #print(self.IC[0,:,:2])
    #print('std:')
    #print(std)
    #print('error_std:')
    #print(error_std)
    #for i, IC in enumerate(self.IC):
        #print(f'{str(self.ens_filter)} exp number {i}')
        #print('mean:')
        #print(IC.mean(axis=-2))
        #print('std:')
        #print(IC.std(axis=-2))
    
    return self.IC

test.build_tests()
gc.collect()
mean=IC_truth+(np.random.normal(size=(n_experiments,1)+IC_truth.shape)*np.sqrt(eigenvalues)*eigenvectors).sum(-1)
build_IC(test.tests[0],eigenvalues, eigenvectors, mean)

test.run()

test.table()

test.plot(ivar=np.s_[:], draw_var=False, draw_std=True, show=True, title='Non-diagonal')

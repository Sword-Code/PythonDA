
import numpy as np
import DA
from Filters import Seik, Ghosh, GhoshV1, GhoshEighT

from scipy.integrate import solve_ivp

#seik=Seik(3)
#ghosh=Ghosh(3, order=5)

#model=DA.Model()
#obs=DA.Observation([0.0, 0.0],[1.0, 1.0])
#test=DA.Experiment([0.0,100.0], model)
#test.build_truth(np.array([1.0, -1.0]))
#test.build_obs(np.arange(1.0,100.0,10.0), obs)

#IC=np.array([[2.0,-2.0],[1.0, -1.0],[3.0, -3.0]])

#test.run(IC, ens_filter=seik)
#test.run(IC, ens_filter=ghosh)
#test.plot()


############

#seik=Seik(2)

#model=DA.Model()
#obs=DA.Observation([0.0],[1.0])
#test=DA.Experiment([0.0,100.0], model)
#test.build_truth(np.array([1.0]))
#test.build_obs(np.arange(1.0, 10.0), obs)

#IC=np.array([[1.0, -1.0],[3.0, -3.0]])

#test.run(IC, ens_filter=seik)
#test.plot()

########################################################################################################

class Lorentz96(DA.Model):
    def __init__(self, F): #, atol=None):
        self.F=F
        #if atol is None:
            #self.atol=1.0e-6
        #else:
            #self.atol=atol
    
    def _L96_flat(self,t,x):
        N,m=self.N,self.m
        d = np.zeros([N,m])
        x_matrix=x.reshape([m,N]).T
        xp1=np.zeros([N,m])
        xp1[0]=x_matrix[-1]
        xp1[1:]=x_matrix[:-1]
        xm2=np.zeros([N,m])
        xm2[:-2]=x_matrix[2:]
        xm2[-2:]=x_matrix[:2]
        xm1=np.zeros([N,m])
        xm1[:-1]=x_matrix[1:]
        xm1[-1]=x_matrix[0]
        d=(xp1-xm2)*xm1-x_matrix+self.F
        return d.T.flatten()
    
    def __call__(self, t_span, state):
        t_span=np.array(t_span)
        state=np.array(state)
        self.N=state.shape[-1]
        x=state.reshape([-1,self.N])
        self.m=x.shape[0]
        x=x.flatten()
        
        sol= solve_ivp(self._L96_flat, t_span, x, dense_output=True)#, atol=self.atol)
        t=sol.t
        y=sol.y.reshape(state.shape+(-1,))
        mysol=DA.MyOdeSolution(sol.sol.ts,sol.sol.interpolants, state.shape)
        
        return y[...,-1], DA.OdeResult(t=t, y=y, sol=mysol)

class First_vars(DA.Observation):
    def __init__(self, obs, std, nvars=3):
        super().__init__(obs, std)
        self.nvars=nvars
        self.indices=np.arange(nvars)
        
    def H(self, state):
        return state[...,:self.nvars]
    
EnsSize=15 #31
N=62 #30
nvars=31 #10
indices=range(0,N,N//nvars)
obs_std=1.0
delta_obs=0.1 #0.2
t_span=[0.0,20.0]
forget=0.8
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

#obs=First_vars(np.zeros(nvars),np.ones(nvars)*obs_std, nvars=nvars)
obs=DA.ObsByIndex(np.zeros(nvars),np.ones(nvars)*obs_std, indices=indices)

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
             #GhoshV1(EnsSize, order=5, forget=forget),
             #GhoshEighT(EnsSize, order=5, forget=forget),
             ]

metrics=[DA.RmpeByTime(index= None, name='RmseTotByTime'),
         DA.RmpeByTime(index= indices, name='RmseObseervedByTime'),
         DA.RmpeByTime(index= tuple(set(range(N))-set(indices)), name='RmseNotObservedByTime'),
        DA.HalfTimeMean(DA.RmpeByTime(index= None), name='RmseTot'),
        DA.HalfTimeMean(DA.RmpeByTime(index= indices), name='RmseObserved'),
        DA.HalfTimeMean(DA.RmpeByTime(index= tuple(set(range(N))-set(indices))), name='RmseNotObserved'),
        ]

model=DA.Lorentz96(F=8) #, atol=atol)

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


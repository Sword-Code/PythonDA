
import numpy as np
import DA
from Filters import Seik, Ghosh, GhoshV1, GhoshV2

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
    def __init__(self, F):
        self.F=F
    
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
        
        sol= solve_ivp(self._L96_flat, t_span, x, dense_output=True)
        t=sol.t
        y=sol.y.reshape(state.shape+(-1,))
        mysol=DA.MyOdeSolution(sol.sol.ts,sol.sol.interpolants, state.shape)
        
        return y[...,-1], DA.OdeResult(t=t, y=y, sol=mysol)

class First_vars(DA.Observation):
    def __init__(self, obs, std, nvars=3):
        super().__init__(obs, std)
        self.nvars=nvars
        
    def H(self, state):
        return state[...,:self.nvars]
    
EnsSize=31
N=30
nvars=3
obs_std=1.0
delta_obs=0.2
t_span=[0.0,10.0]
error_std=np.exp(-np.arange(N))*1.0

ens_filters=[Seik(EnsSize, forget=1.0),
             Ghosh(EnsSize, order=5, forget=1.0),
             GhoshV1(EnsSize, order=5, forget=1.0),
             GhoshV2(EnsSize, order=5, forget=1.0)
             ]

model=Lorentz96(F=8)

obs=First_vars(np.zeros(nvars),np.ones(nvars)*obs_std, nvars=nvars)

test=DA.TwinExperiment(t_span, model, ens_filters)

IC_truth,_ =model([0.0,10.0],[0.01]+[0.0]*(N-1))
test.build_truth(IC_truth)
test.build_obs(np.arange(t_span[0]+delta_obs,t_span[1],delta_obs), obs)
test.build_tests()
test.build_ICs(error_std)

test.run()

test.plot(ivar=0)
test.plot(ivar=4)
test.plot(ivar=15)


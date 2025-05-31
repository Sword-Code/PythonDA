from warnings import warn

import numpy as np
from scipy.integrate._ivp.base import ConstantDenseOutput
from scipy.integrate._ivp.ivp import OdeResult
from scipy.integrate import solve_ivp
from scipy.signal import fftconvolve

from .DA import MyOdeSolution
from .Models import Model 

NO_PLT=False
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as err:
    NO_PLT=True

class LotkaVolterra(Model):
    def __init__(self, N):
        super().__init__()
        self.N=N
        
    def before_transform(self, x):
        return x
    
    def after_transform(self, y):
        return y
    
    def _l_v(self,t,x_flat):
        
        state=x_flat.reshape([self.m,self.state_N])
        x=state[:,None,:self.N]
        b=state[:,None,self.N:2*self.N]
        w=state[:,2*self.N:(self.N+2)*self.N].reshape((self.m, self.N, self.N))
        d=np.zeros(state.shape)
        d[:,None,:self.N]=(b + np.matmul(x, w))*x
        
        return d.flatten()
    
    def __call__(self, t_span, state):
        t_span=np.array(t_span)
        state=np.array(state)
        self.state_N=state.shape[-1]
        x=state.reshape([-1,self.state_N])
        self.m=x.shape[0]
        x=self.before_transform(x)
        sol= solve_ivp(self._l_v, t_span, x.flatten(), dense_output=True)
        
        t=sol.t
        y=sol.y.reshape(state.shape+(-1,))
        y=self.after_transform(y)
        mysol=MyOdeSolution(sol.sol.ts,sol.sol.interpolants, state.shape, after_transform=self.after_transform)
        
        return y[...,-1], OdeResult(t=t, y=y, sol=mysol)
        
    
    def climatological_moments(self, init_IC, init_time=20.0, history_len=1000.0, delta=0.1):
        raise NotImplementedError
        if type(init_IC)==type(1):
            init_IC=np.array([0.01]+[0.0]*(init_IC-1))
        IC_truth,_ =self([0.0,init_time],init_IC)
        IC_truth,history=self([0.0,history_len],IC_truth)
        samples=history.sol(np.arange(0.0,history_len,delta))
        self.clim_mean=samples.mean(-1)
        print("climatological mean:")
        print(self.clim_mean)
        
        cov=np.cov(samples)
        print("climatological std:")
        print(np.sqrt(np.diagonal(cov)))
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        self.clim_eigenvalues=eigenvalues[::-1]
        print("climatological covariance eigenvalues:")
        print(self.clim_eigenvalues)
        self.clim_eigenvectors=eigenvectors[:,::-1]
        
        return IC_truth
    
    def plot(self, x, indices=(0,1,2)):
        if NO_PLT:
            warn("Missing matplotlib module: the plot method cannot be used.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot(x[indices[0]], x[indices[1]], x[indices[2]])
        ax.set_xlabel(f"$x_{indices[0]}$")
        ax.set_ylabel(f"$x_{indices[1]}$")
        ax.set_zlabel(f"$x_{indices[2]}$")
        plt.show()
        
    
class LogLotkaVolterra(LotkaVolterra):
    
    def before_transform(self, x):
        x=x.copy()
        x[...,:self.N]=np.exp(x[...,:self.N])
        return x
    
    def after_transform(self, y):
        y=y.copy()
        y[...,:self.N,:]=np.log(y[...,:self.N,:])
        return y
    

import numpy as np
from scipy.integrate._ivp.base import ConstantDenseOutput
from scipy.integrate._ivp.ivp import OdeResult
from scipy.integrate import solve_ivp
from scipy.signal import fftconvolve
from DA import MyOdeSolution
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.clim_mean=None
        self.clim_eigenvalues=None
        self.clim_eigenvectors=None
    
    def __call__(self, t_span, state):
        return state, OdeResult(t=np.array(t_span), y=np.stack([state]*2, axis=-1), sol=MyOdeSolution(t_span,[ConstantDenseOutput(t_span[0], t_span[1], state.flatten().copy())], state.shape))
    
    def climatological_moments(self, *args, **kwargs):
        raise NotImplementedError
    
class Lorenz96(Model):
    def __init__(self, F=8): #, atol=None):
        super().__init__()
        self.F=F
        #if atol is None:
            #self.atol=1.0e-6
        #else:
            #self.atol=atol
    
    def _L96_flat(self,t,x_flat):
        N,m=self.N,self.m
        d = np.zeros([N,m])
        x=x_flat.reshape([m,N]).T
        
        xm1=np.zeros([N,m])
        xm1[0]=x[-1]
        xm1[1:]=x[:-1]
        
        xm2=np.zeros([N,m])
        xm2[:2]=x[-2:]
        xm2[2:]=x[:-2]
        
        xp1=np.zeros([N,m])
        xp1[:-1]=x[1:]
        xp1[-1]=x[0]
        
        d=(xp1-xm2)*xm1-x+self.F
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
        mysol=MyOdeSolution(sol.sol.ts,sol.sol.interpolants, state.shape)
        
        return y[...,-1], OdeResult(t=t, y=y, sol=mysol)
    
    def climatological_moments(self, init_IC, init_time=20.0, history_len=1000.0, delta=0.1):
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
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot(x[indices[0]], x[indices[1]], x[indices[2]])
        ax.set_xlabel(f"$x_{indices[0]}$")
        ax.set_ylabel(f"$x_{indices[1]}$")
        ax.set_zlabel(f"$x_{indices[2]}$")
        plt.show()
    
class Lorenz05(Lorenz96):
    """
    Lorenz 2005 model, and its two-scale counterparts.
    
    Single scale:
    dX_i = [X,X]_{K,i} - X_i + F
    
    2-scale:
    dZ_i = [X,X]_{K,i} + b^2 (-Y_{i-2}Y_{i-1} + Y_{i-1}Y_{i+1}) +
            + c (-Y_{i-2}X_{i-1} + Y_{i-1}X_{i+1}) - X_i - b Y_i + F
            
    where:
    [X,X]_{K,i} = -W_{i-2K}W_{i-K} + S_{j=-(K/2)}^{K/2} W_{i-K+j}X_{i+K+j}/K,
    W_i = S_{j=-(K/2)}^{K/2} X_{i-j}/K,
    S is:
        a sum where the first and last term are divided by 2, if K is odd;
        the usual sum, if K is even.
        
    In the 2-scale version, Z is the integrated variable, while X and Y are defined as:
        X_i = S_{j= -J}^{J} a_j Z_{i+j},
        Y_i = Z_i - X_i,
        a_j = alpha - beta |j|,
        alpha = (3J^2 + 3)/(2J^3 + 4J),
        beta  = (2J^2 + 1)/(1J^4 + 2J^2).
        
    Parameters:
        F = forcing,
        K = spacial continuty,
        J = smooth steps,
        b = space-time scale
        c = coupling
    """
    
    def __init__(self, F=15, K=None, J=None, two_scale=False, b=10.0, c=3.0): #, atol=None):
        super().__init__(F)
        #if atol is None:
            #self.atol=1.0e-6
        #else:
            #self.atol=atol
        self.K=K
        self.J=J
        self.two_scale=two_scale
        self.b=b
        self.c=c
        
    
    def _L05(self,t,x_flat):
        N,m = self.N,self.m
        d = np.zeros([N,m])
        x=x_flat.reshape([m,N]).T
        d=self.bra(x)-x+self.F
        return d.T.flatten()
    
    def _L05_twoscale(self,t,x_flat):
        N,m = self.N,self.m
        
        d = np.zeros([N,m])
        x=x_flat.reshape([m,N]).T
        
        X=fftconvolve(np.concatenate((x[N-self.J:],x,x[:self.J])), self.a[::-1, None], mode='valid', axes=0)
        Y=x-X
        
        Xp1=np.empty([N,m])
        Xp1[:-1]=X[1:]
        Xp1[-1]=X[0]
        
        Xm1=np.empty([N,m])
        Xm1[0]=X[-1]
        Xm1[1:]=X[:-1]
        
        Yp1=np.empty([N,m])
        Yp1[:-1]=Y[1:]
        Yp1[-1]=Y[0]
        
        Ym1=np.empty([N,m])
        Ym1[0]=Y[-1]
        Ym1[1:]=Y[:-1]
        
        Ym2=np.zeros([N,m])
        Ym2[:2]=Y[-2:]
        Ym2[2:]=Y[:-2]
        
        d=self.bra(X) + self.b**2*Ym1*(Yp1 -Ym2) + self.c*(-Ym2*Xm1 + Ym1*Xp1) - X - self.b*Y + self.F
        
        return d.T.flatten()
    
    def S_even(self,X):
        csum=np.cumsum(X, axis=0)
        array=np.empty([self.N, self.m])
        n=self.N
        array[:self.J+1]=csum[self.J:2*self.J+1]+csum[-1]-csum[-self.J-1:]
        array[self.J+1:n-self.J]=csum[2*self.J+1:]-csum[:-2*self.J-1]
        array[n-self.J:]=csum[-1]-csum[-2*self.J-1:-self.J-1]+csum[:self.J]
        return array
    
    def S_odd(self,X):
        array=self.S_even(X)
        n=self.N
        array[:self.J]-=0.5*(X[self.J:2*self.J]+X[n-self.J:])
        array[self.J:n-self.J]-=0.5*(X[2*self.J:]+X[:-2*self.J])
        array[n-self.J:]-=0.5*(X[-2*self.J:n-self.J]+X[:self.J])
        return array
    
    def W(self, x):
        return self.S(x)/self.K
        
    def bra(self, x):
        array=np.empty([self.N, self.m])
        Wx=self.W(x)
        
        array[:self.K]=Wx[-self.K:]*x[self.K:2*self.K]
        array[self.K:-self.K]=Wx[:-2*self.K]*x[2*self.K:]
        array[-self.K:]=Wx[-2*self.K:-self.K]*x[:self.K]
        array=self.W(array)
        
        array[:self.K]-=Wx[-2*self.K:-self.K]*Wx[-self.K:]
        array[self.K:2*self.K]-=Wx[-self.K:]*Wx[:self.K]
        array[2*self.K:]-=Wx[:-2*self.K]*Wx[self.K:-self.K]
        
        return array
    
    def __call__(self, t_span, state):
        t_span=np.array(t_span)
        state=np.array(state)
        self.N=state.shape[-1]
        x=state.reshape([-1,self.N])
        self.m=x.shape[0]
        x=x.flatten()
        if self.K is None:
            self.K=max([self.N//30,1])
        if self.J is None:
            self.J=self.K//2
        alpha=(3*self.J**2 + 3)/(2*self.J**3 + 4*self.J)
        beta=(2*self.J**2 + 1)/(self.J**4 + 2*self.J**2)
        self.a = alpha - beta * np.abs(np.arange(-self.J,self.J+1))
        if self.K%2==0:
            self.S=self.S_even
        else:
            self.S=self.S_odd
            self.a[0]*=0.5
            self.a[-1]*=0.5
        
        if self.two_scale:
            sol= solve_ivp(self._L05_twoscale, t_span, x, dense_output=True)#, atol=self.atol)
        else:
            sol= solve_ivp(self._L05, t_span, x, dense_output=True)#, atol=self.atol)
        
        t=sol.t
        y=sol.y.reshape(state.shape+(-1,))
        mysol=MyOdeSolution(sol.sol.ts,sol.sol.interpolants, state.shape)
        
        return y[...,-1], OdeResult(t=t, y=y, sol=mysol) 

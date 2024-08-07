import numpy as np
from DA import Observation
import utils
from ._ensfilter import EnsFilter

class Seik(EnsFilter):
    def __init__(self, EnsSize, weights=None, forget=1.0, with_autotuning=False, autotuning_bounds=None):
        super().__init__(EnsSize, weights=weights, forget=forget, with_autotuning=with_autotuning, autotuning_bounds=autotuning_bounds)
        
    def forecast(self, state, Q_std=None, forget=None):
        forget=self._forget_check(forget)
        self.cov1=self.TTW1T*forget[...,None,None]
        if Q_std is not None:
            if np.ndim(Q_std)==0:
                Q_std=np.reshape(Q_std, (1,))
            LT=state[...,1:,:]-np.average(state, weights=self.weights, axis=-2)[...,None,:]
            LTL1=np.linalg.inv(np.matmul(LT, utils.transpose(LT)))
            sqrtQL=np.matmul(LTL1, LT * Q_std[...,None,:])
            cov=np.linalg.inv(self.cov1) + np.matmul(sqrtQL, utils.transpose(sqrtQL))
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            cov1sqrt=eigenvectors/np.sqrt(eigenvalues[...,None,:])
            self.cov1=np.matmul(cov1sqrt,utils.transpose(cov1sqrt))
        
        if state.ndim==2:
            self.cov1=self.cov1.reshape(self.cov1.shape[-2:])
        return state, self._mean_std(state)
    
    def analysis(self, state, obs):
        
        forecast_cov1=self.cov1
        Hstate=obs.H(state)
        
        self._mean_and_base(Hstate)
        self._mean_and_base(state)
        
        #####
        Hstate[...,0,:]=obs.misfit(obs.H(state[...,0,:]))
        #####
        
        sqrtR1HL=obs.sqrtR1(Hstate)
        HLTR1HL=np.matmul(sqrtR1HL,utils.transpose(sqrtR1HL[...,1:,:]))
        cov1=forecast_cov1+HLTR1HL[...,1:,:]
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov1)
        
        change_of_base=utils.transpose(eigenvectors/np.sqrt(eigenvalues[...,None,:]))
        
        state[...,1:,:]=np.matmul(change_of_base,state[...,1:,:])
        
        #####
        base_coef=np.matmul(HLTR1HL[...,0:1,:],utils.transpose(change_of_base))
        #####
        
        state[...,0:1,:]+=np.matmul(base_coef,state[...,1:,:])
        output_state=self.sampling(state)
        
        return output_state, self._mean_std(output_state, mean=state[...,0,:]) 
    
    def sampling(self, mean_and_base):
        omega=np.empty(mean_and_base.shape[:-1]+(self.EnsSize,))
        omega[...,0,:]=self.sqrt_weights
        omega=utils.ortmatrix(omega,1)
        
        change_of_base = omega[...,1:,:]/self.sqrt_weights
        return np.matmul(utils.transpose( change_of_base), mean_and_base[...,1:,:] ) + mean_and_base[...,0:1,:]
    
    def __repr__(self):
        string=f'Seik({self.EnsSize}'
        if self.forget!=1.0:
            string+=f', forget={self.forget}'
        if self.with_autotuning:
            string+=', with autotuning'
        string+=')'
        return string 

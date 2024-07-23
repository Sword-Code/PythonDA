import numpy as np
from DA import Observation
import utils
from ._seik import Seik

class Seik2xSampling(Seik):        
    def forecast(self, state, Q_std=None, forget=None):
        forget=self._forget_check(forget)
        mean=np.average(state,axis=-2,weights=self.weights)
        mean=mean[..., None, :]
        state=(state - mean)/np.sqrt(forget[...,None,None])+mean
        if len(state_shape)==2:
            state=state.reshape(state_shape)
        if Q_std is not None:
            self._mean_and_base(state)
            sqrtQL=state[...,1:,:] / Q_std[...,None,:]
            
            #print('Q_std:')
            #print(Q_std)
            
            LTQ1L=np.matmul(sqrtQL,utils.transpose(sqrtQL))
            cov1=self.TTW1T+LTQ1L
            
            #eigenvalues, eigenvectors = np.linalg.eigh(self.TTW1T)
            #print('eigenvalues TTW1T:')
            #print(eigenvalues) 
            #eigenvalues, eigenvectors = np.linalg.eigh(LTQ1L)
            #print('eigenvalues LTQ1L:')
            #print(eigenvalues) 
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov1)
            
            #print('eigenvalues smoother:')
            #print(eigenvalues) 
            
            #cov1=eigenvectors/np.sqrt(eigenvalues[...,None,:])
            cov1=np.matmul(self.TTW1T,eigenvectors/np.sqrt(eigenvalues[...,None,:]))
            cov1=np.matmul(cov1,utils.transpose(cov1))
            
            #eigenvalues, eigenvectors = np.linalg.eigh(cov1)
            #print('eigenvalues smoother1:')
            #print(eigenvalues) 
            
            #cov1=self.TTW1T-np.matmul(np.matmul(self.TTW1T,cov1),self.TTW1T)
            cov1=self.TTW1T-cov1
            eigenvalues, eigenvectors = np.linalg.eigh(cov1)
            
            #print('eigenvalues cov1:')
            #print(eigenvalues)
            
            change_of_base=utils.transpose(eigenvectors/np.sqrt(eigenvalues[...,None,:]))
            change_of_base=np.matmul(eigenvectors,change_of_base)
            state[...,1:,:]=np.matmul(change_of_base,state[...,1:,:])
            state=self.sampling(state)
        return state, self._mean_std(state)
    
    def __repr__(self):
        string=f'Seik2xSampling({self.EnsSize}'
        if self.forget!=1.0:
            string+=f', forget={self.forget}'
        if self.with_autotuning:
            string+=', with autotuning'
        string+=')'
        return string 
 

import numpy as np
from DA import Observation
import utils
from ._ensfilter import EnsFilter

class Ghosh(EnsFilter):
    def __init__(self, EnsSize, weights_omega=None, forget=1.0, with_autotuning=False, autotuning_bounds=None, order=5, symm=True):
        if weights_omega is None:
            if order<=1:
                raise NotImplementedError
            elif order==2:
                weights=np.ones(EnsSize)/EnsSize
                omega=np.empty((EnsSize,)*2)
                omega[0,:]=np.sqrt(weights)
                self.omega=utils.ortmatrix(omega,1)[1:,:]
                self.symm=False
            elif order==3:
                self.omega=np.identity(self.EnsSize//2)
                weights=np.ones(EnsSize)/EnsSize
                self.symm=True
            elif order==4:
                raise NotImplementedError
            elif order==5:
                hdim=int(np.log2(EnsSize+1))-1
                omega=np.zeros([EnsSize, EnsSize])
                
                omega2=np.zeros([hdim+1,hdim+1])
                for k in range(hdim):
                    omega2[0,k]=2/((k+2)*(k+3))
                    omega2[k+1,k]=k+3
                    omega2[k+1,:k]=1
                omega2[0,hdim]=1-np.sum(omega2[0,:hdim])
                        
                omega[0,0]=np.sqrt(omega2[0,-1])
                for k in range(hdim):
                    omega[0,2**k]=np.sqrt(omega2[0,-2-k]*0.5**k)
                    omega[1:hdim+1,2**k]=np.sqrt(omega2[1:,-2-k])*omega[0,2**k]
                    
                    for k2 in range(k):
                        omega[:hdim+1,2**k+2**k2:2**k+2**(k2+1)]=omega[:hdim+1,2**k:2**k+2**k2]
                        omega[hdim-k2-1,2**k+2**k2:2**k+2**(k2+1)]*=-1
                
                self.omega=omega[1:hdim+1,EnsSize%2:2**hdim]
                weights=np.ones(EnsSize)*(omega2[0,hdim]/(EnsSize-2**(hdim+1)+2))
                weights[1:2**hdim]=omega[0,1:2**hdim]**2*0.5
                weights[EnsSize%2+EnsSize//2:EnsSize//2+2**hdim]=weights[EnsSize%2:2**hdim]
                
                #print('weights:')
                #print(weights)
                #print('omega:')
                #print(self.omega)
                #print('omega^2')
                #print(self.omega @ self.omega.T)
                
                self.symm=True                
            else:
                raise NotImplementedError
            
            
        else:
            raise NotImplementedError
            
        super().__init__(EnsSize, weights=weights, forget=forget, with_autotuning=with_autotuning, autotuning_bounds=autotuning_bounds)
        self.order=order
        
        
    def forecast(self, state, Q_std=None, forget=None):
        forget=self._forget_check(forget)
        mean=np.average(state,axis=-2,weights=self.weights)
        mean=mean[..., None, :]
        state_shape=state.shape
        state=(state - mean)/np.sqrt(forget[...,None,None])+mean
        if len(state_shape)==2:
            state=state.reshape(state_shape)
        if Q_std is not None:
            if np.ndim(Q_std)==0:
                Q_std=np.reshape(Q_std, (1,))
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
    
    def analysis(self, state, obs):
        Hstate=obs.H(state)
        self._mean_and_base(Hstate)
        self._mean_and_base(state)
        
        #####
        Hstate[...,0,:]=obs.misfit(Hstate[...,0,:])
        #####
        
        sqrtR1HL=obs.sqrtR1(Hstate)
        HLTR1HL=np.matmul(sqrtR1HL,utils.transpose(sqrtR1HL[...,1:,:]))
        cov1=self.TTW1T+HLTR1HL[...,1:,:]
        eigenvalues, eigenvectors = np.linalg.eigh(cov1)
        change_of_base=utils.transpose(eigenvectors/np.sqrt(eigenvalues[...,None,:]))
        
        #####
        change_of_base=np.matmul(eigenvectors,change_of_base)
        #####
        
        state[...,1:,:]=np.matmul(change_of_base,state[...,1:,:])
        
        #####
        base_coef=np.matmul(HLTR1HL[...,0:1,:],change_of_base)
        #####
        
        state[...,0:1,:]+=np.matmul(base_coef,state[...,1:,:])
        output_state=self.sampling(state)
        
        #print(self._mean_std(output_state, mean=state[...,0,:]))
        #print(self._mean_std(output_state))
        
        return output_state, self._mean_std(output_state, mean=state[...,0,:])
    
    def sampling(self, mean_and_base):
        #print('presampling')
        #print(np.diag(mean_and_base[1:,:].T @ mean_and_base[1:,:]))
        
        eigenvalues, eigenvectors = np.linalg.eigh(np.matmul(mean_and_base[...,1:,:],utils.transpose(mean_and_base[...,1:,:])))
        #mean_and_base[...,1:,:] = np.matmul(utils.transpose(eigenvectors[...,:,::-1]), mean_and_base[...,1:,:])
        
        omega=np.zeros(mean_and_base.shape[:-1]+(self.EnsSize,))
        omega[...,0,:]=self.sqrt_weights
        hdim,ncol=self.omega.shape
        if self.symm:
            omega2=omega[...,1:1+self.EnsSize//2, self.EnsSize%2 : self.EnsSize%2+self.EnsSize//2]
        else:
            omega2=omega[...,1:,:]
            
        omega3=np.zeros(mean_and_base.shape[:-2]+(hdim,)*2)
        utils.ortmatrix(omega3,0)
        omega2[...,0:hdim,0:ncol]=np.dot(omega3,self.omega)
        
        if self.symm:  
            utils.ortmatrix(omega2,hdim)
            omega2*=np.sqrt(0.5)
            ncol=self.EnsSize//2
            omega[...,1:1+ncol, self.EnsSize%2+ncol:]=-omega2
            
        omega=utils.ortmatrix(omega,ncol+1)
        
        #print('omega:')
        #print(omega)
        #print('omega^2:')
        #print(np.matmul(omega, utils.transpose(omega)))#[omega @ omega.T>1e-5])
        
        change_of_base = np.matmul(eigenvectors[...,:,::-1],omega[...,1:,:] /self.sqrt_weights)
        return np.matmul(utils.transpose( change_of_base), mean_and_base[...,1:,:] ) + mean_and_base[...,0:1,:]
        
    def __repr__(self):
        string=f'Ghosh({self.EnsSize}'
        if self.forget!=1.0:
            string+=f', forget={self.forget}'
        string+=f', order={self.order}'
        if self.with_autotuning:
            string+=', with autotuning'
        string+=')'
        return string 

import numpy as np
from DA import Observation
from DA import utils
from DA import EnsFilter

class Seik(EnsFilter):
    def __init__(self, EnsSize, weights=None, forget=1.0):
        super().__init__(EnsSize, weights, forget)
        self.TTW1T=np.diag(1.0/self.weights[1:])-np.ones([EnsSize-1]*2)
        
    def forecast(self, state, Q_std=None):
        self.cov1=self.TTW1T*self.forget
        if Q_std is not None:
            LT=state[...,1:,:]-np.average(state, weights=self.weights, axis=-2)[...,None,:]
            LTL1=np.linalg.inv(np.matmul(LT, utils.transpose(LT)))
            sqrtQL=np.matmul(LTL1, LT * Q_std[...,None,:])
            cov=np.linalg.inv(self.cov1) + np.matmul(sqrtQL, utils.transpose(sqrtQL))
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            cov1sqrt=eigenvectors/np.sqrt(eigenvalues[...,None,:])
            self.cov1=np.matmul(cov1sqrt,utils.transpose(cov1sqrt))
            
            #print('eigenvalues forecast:')
            #print(eigenvalues)
            
        return state, self._mean_std(state)
    
    def analysis(self, state, obs):
        
        #print('state:')
        #print(state)
        #print('cov:')
        #print(np.cov(state, rowvar=False, aweights=self.weights))
        
        forecast_cov1=self.cov1
        Hstate=obs.H(state)
        
        #print('Hstate:')
        #print(Hstate)
        
        self._mean_and_base(Hstate)
        self._mean_and_base(state)
        
        #print('THstate:')
        #print(Hstate)
        #print('mean and base:')
        #print(state)
        #print('cov:')
        #print(np.matmul(np.matmul(utils.transpose( state[...,1:,:]),np.linalg.inv(self.TTW1T)),state[...,1:,:]))
        
        #####
        Hstate[...,0,:]=obs.misfit(obs.H(state[...,0,:]))
        #####
        
        #print('misfitHstate:')
        #print(Hstate)
        
        sqrtR1HL=obs.sqrtR1(Hstate)
        HLTR1HL=np.matmul(sqrtR1HL,utils.transpose(sqrtR1HL[...,1:,:]))
        cov1=forecast_cov1+HLTR1HL[...,1:,:]
        
        #print('forecast_cov1 and base:')
        #print(forecast_cov1)
        #print('sqrtR1HL:')
        #print(sqrtR1HL)
        #print('HLTR1HL:')
        #print(HLTR1HL)
        #print('cov1:')
        #print(cov1)
        #print('cov:')
        #print(np.matmul(utils.transpose(state[...,1:,:]),np.linalg.inv(cov1)@state[...,1:,:]))
        
        #eigenvalues, eigenvectors = np.linalg.eigh(forecast_cov1)
        #print('eigenvalues forecast 2:')
        #print(eigenvalues)
        #eigenvalues, eigenvectors = np.linalg.eigh(HLTR1HL[...,1:,:])
        #print('eigenvalues HLTR1HL:')
        #print(eigenvalues)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov1)
        
        #print('eigenvalues analisis:')
        #print(eigenvalues)
        
        change_of_base=utils.transpose(eigenvectors/np.sqrt(eigenvalues[...,None,:]))
        
        #print('cov1_eig:')
        #print((eigenvectors*eigenvalues)@eigenvectors.T)
        
        #####
        #change_of_base=matmul(eigenvectors,change_of_base)
        #####
        
        state[...,1:,:]=np.matmul(change_of_base,state[...,1:,:])
        
        #print('cov_2:')
        #print(np.matmul(utils.transpose(state[...,1:,:]),state[...,1:,:]))
        
        #####
        base_coef=np.matmul(HLTR1HL[...,0:1,:],utils.transpose(change_of_base))
        #####
        
        state[...,0:1,:]+=np.matmul(base_coef,state[...,1:,:])
        output_state=self.sampling(state)
        
        #print('output_state:')
        #print(output_state)
        
        return output_state, self._mean_std(output_state, mean=state[...,0,:]) 
    
    def sampling(self, mean_and_base):
        omega=np.empty(mean_and_base.shape[:-1]+(self.EnsSize,))
        omega[...,0,:]=self.sqrt_weights
        omega=utils.ortmatrix(omega,1)
        
        #print('omega:')
        #print(omega)
        #print('omega^2:')
        #print(np.matmul(omega, utils.transpose(omega)))#[omega @ omega.T>1e-5])
        
        change_of_base = omega[...,1:,:]/self.sqrt_weights
        return np.matmul(utils.transpose( change_of_base), mean_and_base[...,1:,:] ) + mean_and_base[...,0:1,:]
    
    def __repr__(self):
        string=f'Seik({self.EnsSize}'
        if self.forget!=1.0:
            string+=f', forget={self.forget}'
        string+=')'
        return string

class Ghosh(EnsFilter):
    def __init__(self, EnsSize, weights_omega=None, forget=1.0, order=None, symm=True):
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
            
        super().__init__(EnsSize, weights, forget)
        self.TTW1T=np.diag(1.0/self.weights[1:])-np.ones([EnsSize-1]*2)
        self.order=order
        
        
    def forecast(self, state, Q_std=None):
        if self.forget!=1.0:
            mean=np.average(state,axis=-2,weights=self.weights)
            mean=mean[..., None, :]
            state=(state - mean)/np.sqrt(self.forget)+mean
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
        string+=')'
        return string
        
        
class GhoshV1(Ghosh):    
    def analysis(self, state, obs):
        Hstate=obs.H(state)
        self._mean_and_base(Hstate)
        self._mean_and_base(state)
        
        #####
        Hstate[...,0,:]=obs.misfit(obs.H(state[...,0,:]))
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
    
    def __repr__(self):
        string=super().__repr__()
        return 'GhoshV1'+ string[5:]
    
class GhoshEighT(Ghosh):
    def _mean_and_base(self, ensemble):   
        mean=np.average(ensemble,axis=-2,weights=self.weights)
        ensemble[...,1:,:]=np.matmul(self.T,(ensemble-ensemble[...,0:1,:]))
        ensemble[...,0,:]=mean
        return ensemble 
    
    def forecast(self, state, Q_std=None):
        if self.forget!=1.0:
            mean=np.average(state,axis=-2,weights=self.weights)
            mean=mean[..., None, :]
            state=(state - mean)/np.sqrt(self.forget)+mean
        if not Q_std is None:
            TTW1T=np.identity(self.EnsSize-1)
            temp=np.average(state, axis=-2,weights=self.weights)
            temp=(state-temp[...,None,:])*self.sqrt_weights[:,None]
            eigenvalues, eigenvectors = np.linalg.eigh(np.matmul(temp,utils.transpose(temp)))
            self.T=utils.transpose(eigenvectors[...,:,-1:0:-1])*self.sqrt_weights
        
            self._mean_and_base(state)
            sqrtQL=state[...,1:,:] / Q_std[...,None,:]
            LTQ1L=np.matmul(sqrtQL,utils.transpose(sqrtQL))
            cov1=TTW1T+LTQ1L
            eigenvalues, eigenvectors = np.linalg.eigh(cov1)
            cov1=utils.transpose(eigenvectors/eigenvalues[...,None,:])
            cov1=np.matmul(eigenvectors,cov1)
            cov1=np.matmul(np.matmul(TTW1T,cov1),LTQ1L)
            eigenvalues, eigenvectors = np.linalg.eigh(cov1)           
            change_of_base=utils.transpose(eigenvectors/np.sqrt(eigenvalues[...,None,:]))
            change_of_base=np.matmul(eigenvectors,change_of_base)
            state[...,1:,:]=np.matmul(change_of_base,state[...,1:,:])
            state=self.sampling(state)
        return state, self._mean_std(state)
    
    def analysis(self, state, obs):
        Hstate=obs.H(state)
        
        temp=np.average(state, axis=-2,weights=self.weights)
        temp=(state-temp[...,None,:])*self.sqrt_weights[:,None]
        eigenvalues, eigenvectors = np.linalg.eigh(np.matmul(temp,utils.transpose(temp)))
        self.T=utils.transpose(eigenvectors[...,:,-1:0:-1])*self.sqrt_weights
        
        self._mean_and_base(Hstate)
        self._mean_and_base(state)
        
        #####
        Hstate[...,0,:]=obs.misfit(Hstate[...,0,:])
        #####
        
        sqrtR1HL=obs.sqrtR1(Hstate)
        HLTR1HL=np.matmul(sqrtR1HL,utils.transpose(sqrtR1HL[...,1:,:]))
        cov1=np.identity(self.EnsSize-1)+HLTR1HL[...,1:,:]
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
        
    def __repr__(self):
        string=super().__repr__()
        return 'GhoshEighT'+ string[5:]
        
    
class NewGhosh(Ghosh):
    def __init__(self, EnsSize, weights_omega=None, forget=1.0, order=None, symm=True):
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
            
        super().__init__(EnsSize, weights, forget)
        self.TTW1T=np.diag(1.0/self.weights[1:])-np.ones([EnsSize-1]*2)
        self.order=order
        
        
    def forecast(self, state, Q_std=None):    
        self.evolved_state=state.copy()
        
        if self.forget!=1.0:
            mean=np.average(state,axis=-2,weights=self.weights)
            mean=mean[..., None, :]
            state=(state - mean)/np.sqrt(self.forget)+mean
        if not Q_std is None:
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
            
            
        if self.last_phase=='a':
            self.take_state=self.analysis_state
        elif self.last_phase=='f':
            self.take_state=self.forecast_state
        else:
            self.take_state=None
        
        self.forecast_state=state.copy()
        self.last_phase='f'
            
        return state, self._mean_std(state)
    
    def analysis(self, state, obs):
        if self.take_state is None:
            pass #matrice e identiata
        else:
            preHstate=obs.H(self.take_state)
            #postHstate=obs.H(self.evolved_state)
            #anomalies=self._mean_and_base(postHstate-preHstate)[...,1:,:]
            #eigenvalues, eigenvectors = np.linalg.eigh(np.matmul(anomalies,utils.transpose(anomalies)))
            #eigenvalues[eigenvalues<0.0]=0.0
            #f=lambda l:
        
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
        
        self.analysis_state=output_state.copy()
        self.last_phase='a'
        
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
        string=super().__repr__()
        return 'NewGosh'+ string[5:]
    

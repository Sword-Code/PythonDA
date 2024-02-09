import numpy as np
import utils

class Metric:
    def __init__(self, delta=0, draw=False, name=None):
        self.delta= delta
        self.draw=draw
        self.name=name
        
    def __call__(self, result, reference, **kwargs):
        if self.delta:
            raise NotImplementedError
        else:
            temp=self._call(result, reference, **kwargs)
            self.result=temp.mean(axis=tuple(range(temp.ndim-2)))
            
            #print(self.name)
            #print(temp.shape)
            
            return self.result
        
    def _call(self, result, reference, **kwargs):
        raise NotImplementedError
    
    def __str__(self):
        if self.name is None:
            return self.__repr__()
        else:
            return self.name
        
    def __repr__(self):
        return 'Metric()'
    
class DistanceByTime(Metric):
    def __init__(self, delta=0, draw=True, name=None):
        super().__init__(delta, draw, name)
        
    def _call(self, result, reference, **kwargs):
        distance=np.abs(reference.sol(result.t ) - result.pre['mean'])
        return distance
    
    def __repr__(self):
        return 'DistanceByTime()'
    
class RmpeByTime(Metric):
    def __init__(self, p=2, index= None, delta=0, draw=True, name=None):
        super().__init__(delta, draw, name)
        self.p=p
        self.index=index
        
    def _call(self, result, reference, **kwargs):
        distance=np.abs(reference.sol(result.t ) - result.pre['mean'])
        if self.index is None:
            distance[...]=np.mean(distance**self.p, axis=-2, keepdims=True)**(1/self.p)
        else:
            if type(self.index)==type(0):
                self.index=(self.index,)
            distance[...]=np.mean(distance[...,self.index,:]**self.p, axis=-2, keepdims=True)**(1/self.p)
        return distance
    
    def __repr__(self):
        strings=[]
        if self.p!=2:
            strings.append(f'p={self.p}')
        if not self.index is None:
            strings.append(f'index={self.index}')
        return f'RmpeByTime({", ".join(strings)})'
    
class Summary(Metric):
    def __init__(self, metric_by_time, draw=False, name=None):
        self.metric_by_time= metric_by_time
        delta=metric_by_time.delta
        super().__init__(delta, draw, name)
        
    def _call(self, result, reference, **kwargs):
        raise NotImplementedError
    
    def __repr__(self):
        string='Summary(' + str(self.metric_by_time) + ')'
        return string

class TimeMean(Summary):
    def __init__(self, metric_by_time, p=2, draw=False, name=None):
        super().__init__(metric_by_time, draw, name)
        self.p=p
        
    def _call(self, result, reference, **kwargs):
        distance=self.metric_by_time._call(result, reference, **kwargs)
        return np.mean(distance**self.p, axis=-1, keepdims=True)**(1/self.p)
    
    def __repr__(self):
        string='TimeMean('+str(self.metric_by_time)
        if self.p!=2:
            string+=f', p={self.p}'
        string+=')'
        return string
    
class LastTime(Summary):        
    def _call(self, result, reference, **kwargs):
        distance=self.metric_by_time._call(result, reference, **kwargs)
        return distance[...,-1:]
    
    def __repr__(self):
        string='LastTime('+str(self.metric_by_time)+')'
        return string
    
class HalfTimeMean(TimeMean):
    def _call(self, result, reference, **kwargs):
        distance=self.metric_by_time._call(result, reference, **kwargs)
        index=distance.shape[-1]//2
        return np.mean(distance[...,index:]**self.p, axis=-1, keepdims=True)**(1/self.p)
    
    def __repr__(self):
        string='HalfTimeMean('+str(self.metric_by_time)
        if self.p!=2:
            string+=f', p={self.p}'
        string+=')'
        return string 
    
class LikelihoodByTime(Metric):
    def __init__(self, delta=0, draw=True, name=None):
        super().__init__(delta, draw, name)
        
    def _call(self, result, reference, **kwargs):
        likelihood=[]
        for t, obs, state in zip(result.t, result.obs, result.pre['state'].transpose((-1,)+tuple(range(result.pre['state'].ndim-1)) )):
            #print('state')
            #print(state.shape)
            if obs==[]:
                likelihood.append(np.zeros(state.shape[:-2]))
                #print('obs 0')
                #print(likelihood[-1].shape)
                continue
            Hstate=obs.H(state)
            Hstate=kwargs['ens_filter']._mean_and_base(Hstate)
            Hstate[...,0,:]=obs.misfit(Hstate[...,0,:])
            
            if Hstate.shape[-2]<obs.obs.shape[-1]:
                sqrtR1HL=obs.sqrtR1(Hstate)
                HLTR1HL=np.matmul(sqrtR1HL[...,1:,:],utils.transpose(sqrtR1HL[...,1:,:]))
                #sqrtR1HL=obs.sqrtR1(Hstate[...,1:,:])
                #HLTR1HL=np.matmul(sqrtR1HL,utils.transpose(sqrtR1HL))
                temp=kwargs['ens_filter'].TTW1T+HLTR1HL
                eigenvalues, eigenvectors = np.linalg.eigh(temp)
                temp=np.matmul(utils.transpose(eigenvectors/np.sqrt(eigenvalues[...,None,:])),sqrtR1HL[...,1:,:])
                #temp=obs.sqrtR1(temp)
                sqrtRPL1sqrtR=np.eye(obs.obs.shape[-1])-np.matmul(utils.transpose(temp),temp)
                eigenvalues, eigenvectors = np.linalg.eigh(sqrtRPL1sqrtR)
                likelihood.append(-2*(np.log(np.sqrt(eigenvectors)).sum(-1)+np.log(obs.std1).sum(-1)) + np.matmul(utils.transpose(eigenvectors*np.sqrt(eigenvalues[...,None,:]), sqrtR1HL[...,0,:, None])**2).sum((-1,-2)))
            else:
                eigenvalues, eigenvectors = np.linalg.eigh(kwargs['ens_filter'].TTW1T)
                temp=np.matmul(utils.transpose(eigenvectors/np.sqrt(eigenvalues[...,None,:])),Hstate[...,1:,:])
                PL=np.diag(obs.std**2)+np.matmul(utils.transpose(temp),temp)
                eigenvalues, eigenvectors = np.linalg.eigh(PL)
                temp=np.matmul(utils.transpose(eigenvectors/np.sqrt(eigenvalues[...,None,:])),Hstate[...,0,:, None])
                likelihood.append((temp**2).sum((-1,-2))+np.log(np.sqrt(eigenvalues)).sum(-1)*2)
            #print('lh')
            #print(likelihood[-1].shape)
            #exit(0)
                
        return np.repeat(np.stack(likelihood,-1)[...,None,:], state.shape[-1], axis=-2)
    
    def __repr__(self):
        return 'LikelihoodByTime()'

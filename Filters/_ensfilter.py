import numpy as np
import utils
from scipy.optimize import minimize
from copy import deepcopy

class EnsFilter:
    def __init__(self, EnsSize, weights=None, forget=1.0, with_autotuning=False, autotuning_bounds=None):
        self.EnsSize=EnsSize
        if weights is None:
            weights=np.ones(EnsSize)/EnsSize
        self.weights=weights
        self.sqrt_weights=np.sqrt(weights)
        self.forget=forget
        self.TTW1T=np.diag(1.0/self.weights[1:])-np.ones([EnsSize-1]*2)
        self.cov1=self.TTW1T
        self.with_autotuning=with_autotuning
        self.autotuned_forget=np.array([forget])
        if autotuning_bounds is None:
            #self.autotuning_bounds=((0.5,1.0), (-np.log(1.5),np.log(1.5)))
            self.autotuning_bounds=((0.5,1.0), (-0.001,0.001))
    
    def forecast(self, state, Q_std=None, forget=None):
        return state, self._mean_std(state)
    
    def autotuning(self, state, obs, Q_std=None):
        state_list=np.reshape(state, (-1,)+state.shape[-2:])
        if Q_std is not None:
            if np.ndim(Q_std)==0:
                Q_std=np.reshape(Q_std, (1,))
            Q_std_list=np.reshape(Q_std, (-1,Q_std.shape[-1]))
            if len(Q_std_list)==1:
                Q_std_list=np.repeat(Q_std_list, state_list.shape[0], axis=0)
        else:
            Q_std_list=[None]*state_list.shape[0]
        obs_list=np.reshape(obs.obs, (-1,obs.obs.shape[-1]))
        if len(obs_list)==1:
            obs_list=np.repeat(obs_list, state_list.shape[0], axis=0)
        std_list=np.reshape(obs.std, (-1,obs.std.shape[-1]))
        if len(std_list)==1:
            std_list=np.repeat(std_list, state_list.shape[0], axis=0)
        template=deepcopy(obs)
        template.obs=obs_list[0]
        template.std=std_list[0]
        template.std1=1.0/template.std
        template.true_std=None
        observation_list=[template]
        for obs_i, std_i in zip(obs_list[1:], std_list[1:]):
            observation=deepcopy(template)
            observation.obs=obs_i
            observation.std=std_i
            observation.std1=1.0/observation.std
            observation.true_std=None
            observation_list.append(observation)
        forget_list=np.reshape(self.autotuned_forget, (-1,))
        if len(forget_list)==1:
            forget_list=np.repeat(forget_list, state_list.shape[0], axis=0)
        new_forget=np.empty(forget_list.shape)
        std_correction=np.empty(forget_list.shape)
        for i, (state_i, obs_i, Q_std_i, forget_i) in enumerate(zip(state_list, observation_list, Q_std_list, forget_list)):
            sol=minimize(fun=self._autotuning_loss, x0=np.array([forget_i, 0.0]), args=(state_i, obs_i, Q_std_i), bounds=self.autotuning_bounds)
            if not sol.success:
                print(f'Experiment {i}:')
                print(sol)
            new_forget[i], std_correction[i]=sol.x
        self.autotuned_forget=new_forget.reshape(state.shape[:-2])
        obs.std=obs.std*np.exp(std_correction.reshape(state.shape[:-2]+(1,)))
        obs.std1=1.0/obs.std
        return self.forecast(state, Q_std=Q_std)
    
    def analysis(self, state, obs):
        return state, self._mean_std(state)
    
    def sampling(self, mean_and_base):
        raise NotImplementedError
        
    def _mean_and_base(self, ensemble):        
        ensemble[...,0,:]=np.average(ensemble,axis=-2,weights=self.weights)
        ensemble[...,1:,:]-=ensemble[...,0:1,:]        
        return ensemble 
    
    def _mean_std(self, ensemble, mean=None):
        if mean is None:
            
            #print(self.weights)
            #print(ensemble)
            
            mean=np.average(ensemble,axis=-2,weights=self.weights)
        else:
            mean=mean.copy()    
        mean=mean[...,None,:]
        anomalies=ensemble-mean
        std=np.sqrt(np.average(anomalies**2,axis=-2,weights=self.weights))
        mean=mean[...,0,:]  
        #print(std[0,:2])
        return mean, std
    
    def _autotuning_loss(self, x, *args):
        forget, std_correction=x
        state, obs, Q_std=args
        forecast_state, _=self.forecast(state, Q_std=Q_std, forget=forget)
        corrected_obs=deepcopy(obs)
        corrected_obs.std=obs.std*np.exp(std_correction)
        corrected_obs.std1=1.0/corrected_obs.std
        score=utils.log_likelihood(corrected_obs, forecast_state, self.cov1, self._mean_and_base)
        return score
    
    def _forget_check(self,forget):
        if forget is None:
            if self.with_autotuning:
                forget=self.autotuned_forget
            else:
                forget=self.forget
        if np.ndim(forget)==0:
            forget=np.array([forget])
        return forget
    
    def __repr__(self):
        string=f'EnsFilter({self.EnsSize}'
        if self.forget!=1.0:
            string+=f', forget={self.forget}'
        string+=')'
        return string 

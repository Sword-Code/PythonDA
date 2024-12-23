import numpy as np
from scipy.integrate._ivp.base import ConstantDenseOutput, DenseOutput
from scipy.integrate._ivp.ivp import OdeResult
from scipy.integrate import OdeSolution, solve_ivp
import itertools
from copy import deepcopy
from time import time

from Metrics import DistanceByTime, RmpeByTime, LastTime, HalfTimeMean, TimeMean, LastTime, HalfTimeMean, TimeMean
import Metrics
import utils

from warnings import warn

# import matplotlib.pyplot as plt
# from tabulate import tabulate

NO_TABULATE=False
try:
    from tabulate import tabulate
except ModuleNotFoundError as err:
    NO_TABULATE=True
    warn(f"{err}. The TwinExperiment.table method will not be available.")
    
NO_PLT=False
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as err:
    NO_PLT=True
    warn(f"{err}. The TwinExperiment.plot method will not be available.")

class BaseObservation:
    def __init__(self, obs, std, true_std=None):
        self.obs=np.array(obs)
        if np.ndim(self.obs)==0:
            self.obs=np.reshape(self.obs, (1,))
        self.std=np.array(std)
        if np.ndim(self.std)==0:
            self.std=np.reshape(self.std, (1,))
        self.std1=1.0/self.std
        self.true_std=true_std
        if true_std is None:
            self.true_std=self.std
        else:
            if np.ndim(true_std)==0:
                self.true_std=np.reshape(true_std, (1,))
    
    def misfit(self, Hstate):
        return self.obs-Hstate
    
    def sqrtR1(self, Hbase):
        return Hbase*self.std1[...,None,:]
            
    def H(self, state):
        raise NotImplementedError
    
class Observation(BaseObservation):
    def __init__(self, obs, std, true_std=None, indices=None):
        super().__init__(obs, std, true_std=true_std)
        if indices is None:
            self.indices=np.arange(obs.shape[-1])
        else:
            self.indices=indices
        
    def H(self, state):
        return state[...,self.indices]
    
class Multiobs(BaseObservation):
    def __init__(self, observations):
        self.observations=observations
        self.obs=np.concatenate([observation.obs for observation in self.observations], axis=-1)
        self.std=np.concatenate([observation.std for observation in self.observations], axis=-1)
        self.std1=np.concatenate([observation.std1 for observation in self.observations], axis=-1)
        self.true_std=np.concatenate([observation.true_std for observation in self.observations], axis=-1)
        
    def H(self, state):
        return np.concatenate([observation.H(state) for observation in self.observations], axis=-1)
    
    def sqrtR1(self, Hbase):
        return np.concatenate([observation.sqrtR1(Hbase) for observation in self.observations], axis=-1)
    
class MyOdeSolution(OdeSolution):
    def __init__(self,ts, interpolants, shape):
        super().__init__(ts, interpolants)
        self.shape=shape
        
    def __call__(self, t):
        result=super().__call__(t)            
        return result.reshape(self.shape+(-1,)*(result.ndim-1))
    
class Model:
    def __init__(self):
        pass
    
    def __call__(self, t_span, state):
        return state, OdeResult(t=np.array(t_span), y=np.stack([state]*2, axis=-1), sol=MyOdeSolution(t_span,[ConstantDenseOutput(t_span[0], t_span[1], state.flatten().copy())], state.shape))
    
class Lorenz96(Model):
    def __init__(self, F=8): #, atol=None):
        self.F=F
        #if atol is None:
            #self.atol=1.0e-6
        #else:
            #self.atol=atol
    
    def _L96_flat(self,t,x_flat):
        N,m=self.N,self.m
        d = np.zeros([N,m])
        x=x_flat.reshape([m,N]).T
        xp1=np.zeros([N,m])
        xp1[0]=x[-1]
        xp1[1:]=x[:-1]
        xm2=np.zeros([N,m])
        xm2[:-2]=x[2:]
        xm2[-2:]=x[:2]
        xm1=np.zeros([N,m])
        xm1[:-1]=x[1:]
        xm1[-1]=x[0]
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
    
class MyDenseOutput(DenseOutput):
    def __init__(self, interpolant, delta_min, delta_max):
        super().__init__(interpolant.t_old, interpolant.t)
        self.delta_min=delta_min
        self.delta_max=delta_max
        self.interpolant=interpolant
        
    def _call_impl(self, t):
        delta=(t-self.t_min)/(self.t_max-self.t_min)*self.delta_max + (self.t_max-t)/(self.t_max-self.t_min)*self.delta_min
        return self.interpolant(t)+delta        


    
STANDARD_METRICS = [DistanceByTime(), 
                    RmpeByTime(name='RmseByTime'), 
                    LastTime(DistanceByTime()), 
                    HalfTimeMean(DistanceByTime()), 
                    TimeMean(DistanceByTime()), 
                    LastTime(RmpeByTime(name='RmseByTime')), 
                    HalfTimeMean(RmpeByTime(name='RmseByTime')), 
                    TimeMean(RmpeByTime(name='RmseByTime')),
                   ]

class Counter():
    def __init__(self):
        self.clear()
    
    def clear(self):
        for key in self.__dict__:
            self.__dict__[key]=0
        #self.forecast=0
        #self.analysis=0
        #self.model=0
        return self
    
    def add(self, forecast=0, analysis=0, model=0):
        self.forecast+=forecast
        self.analysis+=analysis
        self.model+=model
        return self
    
    def add(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key]+=value
            else:
                self.__dict__[key]=value
        return self
    
    def __add__(self, counter):
        return self.add(counter.forecast, counter.analysis, counter.model)
    
    def __add__(self, counter):
        return self.add(**counter.__dict__)
    
    def __str__(self):
        return f'Counter: forecast={self.forecast}, analysis={self.analysis}, DA={self.forecast+self.analysis}, model={self.model}'
    
    def __str__(self):
        s='Counter: '
        for key, value in self.__dict__.items():
            s+=f'{key}={value}, '
        return s[:-2]
        #return f'Counter: forecast={self.forecast}, analysis={self.analysis}, DA={self.forecast+self.analysis}, model={self.model}'

class Test:
    def __init__(self, t_span, model, ens_filter, observations, IC=None, delta_forecast=0, Q_std_t = None, metrics = STANDARD_METRICS, reference=None, label=None):
                     
        self.model=model
        self.t_span=t_span
        self.IC=IC
        self.observations=observations
        self.ens_filter=ens_filter
        self.delta_forecast=delta_forecast
        if not callable(Q_std_t):
            self.Q_std_t=lambda t: Q_std_t
        else:
            self.Q_std_t=Q_std_t
        if label is None:
            self.label=repr(ens_filter)
        else:
            self.label=label
        self.metrics=metrics
        self.reference=reference
        self.counter=Counter()
        
    def compute_metrics(self, result=None, reference=None):
        if result is None:
            result=self.result
        if reference is None:
            reference=self.reference
        self.metrics_result=[]
        self.draw_metrics=[]
        if reference is None:
            return
        for metric in self.metrics:
            #print(self.ens_filter)
            #print(metric(result, reference))
            metric(result, reference, ens_filter=self.ens_filter)
            if metric.draw:
                self.draw_metrics.append(metric)
            else:
                self.metrics_result.append(metric)
        
    def run(self, with_metrics=True, poly_sol=False):
        IC=self.IC
        t_span=self.t_span
        ens_filter=self.ens_filter
        Q_std_t=self.Q_std_t
        delta=self.delta_forecast
        
        state=IC.copy()
        t=[t_span[0]]
        obs_now=[]
        pre={'state':[], 'mean':[], 'std':[]}
        post={'state':[], 'mean':[], 'std':[]}
        observations=[]
        segments=[]
        sorted_obs=list(self.observations)
        sorted_obs.sort(key=lambda x:x[0])
        self.counter.clear()
        
        for t_obs, obs in sorted_obs+[(t_span[1], None)]:
            #print(t_obs)
            if t_obs<t_span[0]: continue
            if t_obs>t_span[1]: break
            
            if delta:
                repeat=round((t_obs-t[-1])/delta)
            else:
                repeat=int(t_obs>t[-1])
                
            if t_obs<=t[-1]+delta*0.5:
                if obs:
                    obs_now.append(obs)
                    continue
            
            #print('t[-1]:')
            #print(t[-1])
            #print('Q_std_t(t[-1]):')
            #print(Q_std_t(t[-1]))
            
            preforecast=state
            timer=time()
            state, (mean,std) = ens_filter.forecast(state, Q_std_t(t[-1]))
            timer=time()-timer
            self.counter.add(forecast=timer)
            
            pre['state'].append(state.copy())
            pre['mean'].append(mean)
            pre['std'].append(std)
            
            if obs_now:
                all_obs=Multiobs(obs_now)
                obs_now=deepcopy(all_obs)
                if ens_filter.with_autotuning:
                    timer=time()
                    state, _ = ens_filter.autotuning(preforecast, all_obs, Q_std=Q_std_t(t[-1]))
                    timer=time()-timer
                    self.counter.add(autotuning=timer)
                timer=time()
                state, (mean,std) = ens_filter.analysis(state,all_obs)
                timer=time()-timer
                self.counter.add(analysis=timer)
                
            post['state'].append(state.copy())
            post['mean'].append(mean)
            post['std'].append(std)
            observations.append(obs_now)
            
            #print("pre['state']:")
            #print(pre['state'])
            #print('pre[mean]:')
            #print(pre['mean'])
            #print('pre[std]:')
            #print(pre['std'][-1][0, :2])
            #print("post['state']")
            #print(post['state']) 
            #print('post[mean]:')
            #print(post['mean'])
            #print('post[std]:')
            #print(post['std'][-1][0, :2])
            
            for i in range(repeat):
                tf=t[-1]+delta if delta else t_obs
                
                #print('state pre model:')
                #print(state)
                
                timer=time()
                state, full_model_result = self.model([t[-1], tf], state)
                timer=time()-timer
                self.counter.add(model=timer)
                
                #print('state post model:')
                #print(state)
                
                if (i<repeat-1) or (obs is None):
                    obs_now=[]
                else:
                    obs_now=[obs]
                t.append(tf)
                if poly_sol:
                    segments.append(full_model_result)
                if not obs_now:
                    timer=time()
                    state, (mean,std) = ens_filter.forecast(state, Q_std_t(t[-1]))
                    timer=time()-timer
                    self.counter.add(forecast=timer)
                    
                    pre['state'].append(state.copy())
                    pre['mean'].append(mean)
                    pre['std'].append(std)
                    post['state'].append(state.copy())
                    post['mean'].append(mean)
                    post['std'].append(std)
                    observations.append(obs_now)
        
        pre={'state':np.stack(pre['state'],-1), 'mean':np.stack(pre['mean'],-1), 'std':np.stack(pre['std'],-1)}
        post={'state':np.stack(post['state'],-1), 'mean':np.stack(post['mean'],-1), 'std':np.stack(post['std'],-1)}
        if poly_sol:
            sol=MyOdeSolution(np.concatenate([segments[0].sol.ts]+[segment.sol.ts[1:] for segment in segments[1:]]),
                            list(itertools.chain(*[segment.sol.interpolants for segment in segments])),
                            state.shape)
        else:
            sol=None
        result=OdeResult(t=np.array(t), pre=pre, post=post, sol=sol, obs=observations, ens_filter=ens_filter)
        self.result =result
        
        if with_metrics:
            self.compute_metrics()
            
        return result
    
    def build_IC(self, std=1.0, mean=None):
        if mean is None:
            mean=self.reference.y[...,0]
        error_std=mean.copy()
        error_std[...]=std
        
        #print('error_std:')
        #print(error_std[:,:2])
        
        mean_and_base=np.zeros(mean.shape[:-1]+(self.ens_filter.EnsSize,mean.shape[-1]))
        mean_and_base[...,0,:]=mean
        
        #mean_and_base[...,np.arange(1,self.ens_filter.EnsSize),np.arange(self.ens_filter.EnsSize-1)]=error_std*np.sqrt(self.ens_filter.forget)
        
        indices=np.flip(np.argsort(error_std), axis=-1)[...,:self.ens_filter.EnsSize-1]
        adv_slices=np.zeros((indices.ndim,)+ indices.shape, dtype=int)
        for i, temp in enumerate(adv_slices):
            temp[...]=np.arange(indices.shape[i])[np.index_exp[...]+(None,)*(indices.ndim-1-i)]
        temp=mean_and_base[...,1:,:]
        temp[np.index_exp[...]+(*adv_slices, indices)]=np.take_along_axis(error_std, indices, axis=-1)
        
        #print('mean_and_base:')
        #print(mean_and_base[0,:,:2])
        #print(np.matmul(utils.transpose(mean_and_base[...,1:,:]),mean_and_base[...,1:,:]))
        
        self.IC=self.ens_filter.sampling(mean_and_base)
        
        #print('IC:')
        #print(self.IC[0,:,:2])
        #print('std:')
        #print(std)
        #print('error_std:')
        #print(error_std)
        #for i, IC in enumerate(self.IC):
            #print(f'{str(self.ens_filter)} exp number {i}')
            #print('mean:')
            #print(IC.mean(axis=-2))
            #print('std:')
            #print(IC.std(axis=-2))
        
        return self.IC
    
    def build_climatological_IC(self, mean=None):
        if self.model.clim_mean is None:
            self.model.climatological_moments(self.reference.y[...,0].shape[-1])
        if mean is None:
            mean=self.model.clim_mean
        eigenvalues=self.model.clim_eigenvalues
        eigenvectors=self.model.clim_eigenvectors
        
        mean_and_base=np.zeros(mean.shape[:-1]+(self.ens_filter.EnsSize,mean.shape[-1]))
        mean_and_base[...,0,:]=mean
        mean_and_base[...,1:,:]=(eigenvectors[:,:self.ens_filter.EnsSize-1] *np.sqrt(eigenvalues[:self.ens_filter.EnsSize-1])).transpose()
        
        self.IC=self.ens_filter.sampling(mean_and_base)
        
        return self.IC
    
class TwinExperiment:
    def __init__(self, t_span, model, ens_filters, observations=None, IC=None, delta_forecast=0, Q_std_t = None, metrics = STANDARD_METRICS, reference=None):
        self.model=model
        self.t_span=t_span
        self.IC=IC
        self.observations=observations
        self.ens_filters=ens_filters
        self.delta_forecast=delta_forecast
        if not callable(Q_std_t):
            self.Q_std_t=lambda t: Q_std_t
        else:
            self.Q_std_t=Q_std_t
        self.metrics=metrics
        self.reference=reference
        
    def run(self):
        for test in self.tests:
            print('Running '+ str(test.ens_filter) +'...')
            timer=time()
            test.run()
            timer=time()-timer
            print(str(test.ens_filter)+f' done in {timer} seconds.')
            test.counter.DA=test.counter.forecast+test.counter.analysis
            test.counter.total=timer 
            print(test.counter)
            #print(test.)
        
    def build_tests(self):
        self.tests=[]
        for ens_filter in self.ens_filters:
            metrics=deepcopy(self.metrics)
            self.tests.append(Test(self.t_span, self.model, ens_filter, self.observations, self.IC, self.delta_forecast, self.Q_std_t, metrics, self.reference))
        return self.tests
    
    def build_truth(self, IC, t_span=None, delta=0, Q_std=None):
        if t_span is None:
            t_span=self.t_span
        if delta==0 or Q_std is None:            
            _, self.reference = self.model(t_span, IC)
        else:
            t0=t_span[0]
            state=IC
            segments=[]
            for t in np.arange(t0+delta,t_span[1]+delta*0.5,delta):
                state, segment = self.model([t0,t], state)
                error=np.random.normal(size=state.shape)*Q_std
                segments.append((error,segment))
                t0=t
                state=state+error
            t=np.concatenate([segments[0][1].t]+[segment[1].t[1:] for segment in segments[1:]])
            y=np.concatenate([segments[0][1].y + segments[0][0][...,None]*(segments[0][1].t-segments[0][1].t[0])/delta] + [segment[1].y[...,1:] + segment[0][...,None]*(segment[1].t[1:]-segment[1].t[0])/delta for segment in segments[1:]], axis=-1)
            
            my_interpolants=[]
            for seg_error, seg_sol in segments:
                for interpolant in seg_sol.sol.interpolants:
                    my_interpolants.append(MyDenseOutput(interpolant, 
                                                         delta_min = seg_error.flatten()/delta*(interpolant.t_min-seg_sol.t[0]),
                                                         delta_max = seg_error.flatten()/delta*(interpolant.t_max-seg_sol.t[0])))
            
            sol=MyOdeSolution(np.concatenate([segments[0][1].sol.ts]+[segment[1].sol.ts[1:] for segment in segments[1:]]),
                              my_interpolants,
                              IC.shape)
            self.reference=OdeResult(t=t, y=y, sol=sol)
        return self.reference
    
    def build_obs(self, times, template, reference=None, true_std_sigma=0):
        if reference is None:
            reference=self.reference
        observations=[]
        for t in times:
            if t<reference.t[0] or t>reference.t[-1]:
                continue
            obs=deepcopy(template)
            obs.true_std=template.std*np.exp(np.random.normal(size=template.true_std.shape)*np.abs(true_std_sigma))
            error=np.random.normal(size=obs.std.shape)*obs.true_std
            obs.obs=obs.H(reference.sol(t))+error
            observations.append((t, obs))
        self.observations=observations
        
        return observations
    
    def build_ICs(self, std=1.0, truth_0=None, n_experiments=1):
        if truth_0 is None:
            truth_0=self.reference.y[...,0]
        error_std=truth_0.copy()
        error_std[...]=std
        
        mean=truth_0+np.random.normal(size=(n_experiments,)+truth_0.shape)*error_std
        #print('truth_0:')
        #print(truth_0)
        #print('mean:')
        #print(mean)
        #mean=truth_0+np.random.normal(size=truth_0.shape)*error_std
        #print(mean)
        for test in self.tests:
            test.build_IC(std=error_std, mean=mean)
            
        return mean
    
    def build_climatological_ICs(self, truth_0=None, n_experiments=1):
        if truth_0 is None:
            truth_0=self.reference.y[...,0]
        if self.model.clim_eigenvalues is None:
            self.model.climatological_moments(truth_0.shape[-1])
        
        mean=truth_0+(np.random.normal(size=(n_experiments,1)+truth_0.shape)*np.sqrt(self.model.clim_eigenvalues)*self.model.clim_eigenvectors).sum(-1)
        
        for test in self.tests:
            test.build_climatological_IC(mean=mean)
            
        return mean
            
    def table(self, ivar=None):
        if NO_TABULATE:
            warn("The 'table' method needs the tabulate module.")
            return
        array=[[ variable,  "\n".join([str(metric) for metric in self.tests[0].metrics_result])] + 
               ["\n".join([str(metric.result[variable].item()) for metric in test.metrics_result]) for test in self.tests] for variable in range(len(self.tests[0].metrics_result[0].result))]
        headers=['Variable', 'Metric']+[str(test.ens_filter) for test in self.tests]
        print(tabulate(array,headers=headers, tablefmt="fancy_grid"))
        
    
    def plot(self, ivar=0, iexp=np.s_[:], draw_var=True, draw_std=True, draw_metrics=True, draw_ens=False, show=True, save=None, title=None):
        if NO_PLT:
            warn("The 'plot' method needs the matplotlib module.")
            return
        fig, ax_list = plt.subplots(int(draw_var)+int(draw_std)+int(draw_metrics)*len(self.tests[0].draw_metrics), sharex=True, squeeze=False, figsize=[12.8,4.8+4.8*int(draw_metrics)])
        ax_list=ax_list.flatten()
        
        if draw_var:
            ax=ax_list[0]
            ax.plot(self.reference.t, self.reference.y[ivar], 'k', label='Truth')
            not_drawable_obs=False
            for iobs, (t, obs) in enumerate(self.observations):
                if 'indices' not in obs.__dict__:
                    all_obs_drawable=True
                    continue
                if ivar in obs.indices:
                    i=tuple(obs.indices).index(ivar)
                    obs_value=obs.obs[...,i]
                    if obs_value.ndim > 0:
                        obs_value=obs_value[iexp]
                    if obs_value.ndim > 0:
                        obs_value=obs_value.mean()
                    obs_line,=ax.plot([t],obs_value, 'go')
                    if iobs==0:
                        obs_line.set_label('Observations')
            if not_drawable_obs:
                warn("Some observations are not drawable: missing 'indices' attribute. Probably you are not using the Observation class. ")
        for itest, test in enumerate(self.tests):
            color=f'C{itest % 10}'
            ax_number=-1
            
            if draw_var:
                ax_number+=1
                ax=ax_list[ax_number]
                
                time=np.repeat(test.result.t,2)
                
                array=np.stack([test.result.pre['mean'][..., iexp, ivar,:],test.result.post['mean'][...,iexp, ivar,:]],axis=-1)
                if array.ndim>=2:
                    array=array.mean(tuple(range(array.ndim-2)))
                
                #print('plot mean:')
                #print(array)
                
                line,=ax.plot(time, array.flatten(), label=test.label)
                
                array_u=np.stack([test.result.pre['mean'][...,iexp, ivar,:]+test.result.pre['std'][...,iexp, ivar,:], test.result.post['mean'][...,iexp, ivar,:]+test.result.post['std'][...,iexp, ivar,:]],axis=-1)
                if array_u.ndim>=2:
                    array_u=array_u.mean(tuple(range(array_u.ndim-2)))
                #ax.plot(time, array_u.flatten(), '--', color=color, label=test.label+' std')
                
                array_d=np.stack([test.result.pre['mean'][...,iexp, ivar,:]-test.result.pre['std'][...,iexp, ivar,:], test.result.post['mean'][...,iexp, ivar,:]-test.result.post['std'][...,iexp, ivar,:]],axis=-1)
                if array_d.ndim>=2:
                    array_d=array_d.mean(tuple(range(array_d.ndim-2)))
                #ax.plot(time, array_d.flatten(), '--', color=color)
                
                ax.fill_between(time, array_u.flatten(), array_d.flatten(), alpha=0.5, label=test.label+' std')
                
                if draw_ens: #to be correct after adding iexp
                    for member_pre, member_post in zip(test.result.pre['state'].reshape((-1,)+test.result.pre['state'].shape[-2:]),test.result.post['state'].reshape((-1,)+test.result.pre['state'].shape[-2:])):
                        member_line,=ax.plot(time, np.stack([member_pre[ivar],member_post[ivar]],axis=-1).flatten(), ':', color=color, alpha=0.2)
                    member_line.set_label(test.label + ' ensemble')
                
            if draw_std:
                ax_number+=1
                ax=ax_list[ax_number]
                
                time=np.repeat(test.result.t,2)
                
                array=np.stack([test.result.pre['std'][...,iexp, ivar,:],test.result.post['std'][...,iexp, ivar,:]],axis=-1)
                if array.ndim>=2:
                    array=np.sqrt((array**2).mean(tuple(range(array.ndim-2))))
                ax.plot(time, array.flatten(), color=color, label=test.label)
            
            if draw_metrics:
                for metric in test.draw_metrics:
                    ax_number+=1
                    ax=ax_list[ax_number]
                    
                    #print(str(metric))
                    #print(metric.result[ivar])
                    #print(metric)
                    #print(test.result.t.shape)
                    #print(metric.result[ivar].shape)
                    
                    array=metric.result[ivar]
                    if array.ndim>=1:
                        array=array.mean(tuple(range(array.ndim-1)))
                    
                    ax.plot(test.result.t, array, color=color, label=test.label)
        
        ax_number=-1
        
        if draw_var:
            ax_number+=1
            ax=ax_list[ax_number]
            ax.set_title(f'Variable {ivar}')
            ax.set_xlabel('Time')
            ax.set_ylabel(f'y[{ivar}]')
            ax.legend()
        
        if draw_std:
            ax_number+=1
            ax=ax_list[ax_number]
            ax.set_title(f'STD Variable {ivar}')
            ax.set_xlabel('Time')
            ax.set_ylabel('y')
            ax.set_ylim(bottom=0.0)
            ax.legend()
        
        if draw_metrics:
            for metric in self.tests[0].draw_metrics:
                ax_number+=1
                ax=ax_list[ax_number]
                ax.set_title(str(metric))
                ax.set_xlabel('Time')
                ax.set_ylabel(str(metric))
                ax.set_ylim(bottom=min([0.0, ax.get_ylim()[0]]))
                ax.legend()
        if title is not None:
            fig.suptitle(title)#, fontsize=16)
        
        fig.tight_layout()
        #ax.plot(time, [1/np.sqrt(i*0.5) for i in range(1,len(time)+1)], 'b--')
        
        #print(time)
        #print(self.result.pre['mean'])
        #print(self.result.post['mean'])
        #print(self.result.pre['state'])
        #print(self.result.post['state'])
        #print(np.stack([self.result.pre['mean'],self.result.post['mean']],axis=-1).flatten())
        
        if save is not None:
            fig.savefig(save)
        
        if show:
            plt.show()
        
    
        
        


# PythonDA

This is a Python Data Assimilation tool. It contains classes for DA experiments, twin experiments, DA filters and metrics. 

## Usage:

Prepare a list of ensemble filters for a twin experiment:
```
from Filters import Seik, Ghosh

EnsSize=31
forgettin_factor=0.7
ens_filters=[Seik(EnsSize, forget=forgettin_factor), 
             Ghosh(EnsSize, forget=forgettin_factor))]
```
Each filter needs to know the ensemble size. Other customizations are possible by specifying other arguments (e.g., the chosen forgetting factor, which is 1.0 by default).

Choose a model:
```
import Models

model=Models.Lorenz96()
```

Choose metrics to be compared:
```
import Metrics

metrics=[Metrics.RmpeByTime(name='RMSE_by_time'),
         Metrics.TimeMean(Metrics.RmpeByTime(), name='Total_RMSE')]
```
The first metric `RmpeByTime` is a metric over time, suitable to be represented in a plot. It computes the p-norm of the error (p is 2 by defalut, giving the Root Mean Square Error, but it can be specified as argument, e.g., `Metrics.RmpeByTime(p=1)`). The second one, `TimeMean`, takes a metric over time as argument (e.g., `RmpeByTime`), and computes its p-norm in the time window. `TimeMean` is a subclass of `Summary` metrics, which are suitable to summarize the results with a single number and usually are represented in tables better than plots.

Define a twin experiment in the time window between 0 and 20:
```
import DA

t_span=[0.0,20.0]
test=DA.TwinExperiment(t_span=t_span, model=model, ens_filters=ens_filters, metrics = metrics)
```

Compute climatlogical moments (mean and covariance) for the model and build the truth of the twin experiment:
```
N=62
IC_truth=model.climatological_moments(N)
test.build_truth(IC_truth)
```
Here `N` is the number of variables of the Lorenz96 model. 
The `climatological_moments` method:
1. build an history by integrating the model for a long period (default is `history_len=1000.0`),
2. computes climatological mean and covariance,
3. returns the last state of the history.
The `build_truth` method create the truth of the twin experiment integrating the model in the time window `test.t_span`, with initial condition `IC_truth`.

Build observations:
```
import numpy as np

obs_template=DA.Observation(obs=np.zeros(N), std=np.ones(N))
obs_times=np.arange(t_span[0]+0.2,t_span[1], 0.2)
test.build_obs(times=obs_times, template=obs_template)
```
The observations in the twin experiment are built given an observation template and observation times.

Initialize experiments, build error affected initial conditions and launch the twin experiments:
```
test.build_tests()
n_experiments=100
test.build_climatological_ICs(n_experiments=n_experiments)
test.run()
```


Present results by using a table and a plot:
```
test.table()
test.plot()
```

See examples in single_twin.py and multi_twin.py.

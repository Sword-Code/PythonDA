# PythonDA

This is a Python Data Assimilation tool. It contains classes for DA experiments, twin experiments, DA filters and metrics. 

## Usage:

Prepare a list of ensemble filters for a twin experiment:
```
from Filters import Seik, Ghosh

EnsSize=31
ens_filters=[Seik(EnsSize), Ghosh(EnsSize)]
```

Choose a model:
```
import DA

model=DA.Lorenz96()
```

Choose metrics to be compared:
```
metrics=[DA.RmpeByTime(name='RMSE_by_time'),
         DA.TimeMean(DA.RmpeByTime(), name='Total_RMSE')]
```
The first metric `RmpeByTime` is a metric over time, suitable to be represented in a plot. The second one, `TimeMean`, takes a metric over time as argument (e.g., `RmpeByTime`), and computes its p-norm (p is 2 by defalut, but can be specified as argument, e.g., `TimeMean(DA.RmpeByTime(), p=1)`). `TimeMean` is a subclass of `Summary` metrics, which are suitable to summarize the results with a single number and usually are represented in tables better than plots.

Define a twin experiment in the time window betweein 0 and 20:
```
t_span=[0.0,20.0]
test=DA.TwinExperiment(t_span=t_span, model=model, ens_filters=ens_filters, metrics = metrics)
```

Build the truth:
```
N=62
IC_truth,_ =model([0.0,20.0],[0.01]+[0.0]*(N-1))
test.build_truth(IC_truth)
```
Here `N` is the number of variables of the Lorenz96 model. The initial condition for the truth is computed integrating the model for 20 time units after adding, at the zero state, a small perturbation (0.01) to the first variable.

Build observations:
```
import numpy as np

obs=DA.Observation(obs=np.zeros(N), std=np.ones(N))
obs_times=np.arange(t_span[0]+0.2,t_span[1], 0.2)
test.build_obs(times=obs_times, template=obs)
```

Initialize experiments, build error affected initial conditions and launch the twin experiment:
```
test.build_tests()
error_std=np.ones(N)*5.0
test.build_ICs(error_std, n_experiments=100)
test.run()
```

Present results by using a table and a plot:
```
test.table()
test.plot()
```

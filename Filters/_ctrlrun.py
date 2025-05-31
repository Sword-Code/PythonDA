import numpy as np
from ._ensfilter import EnsFilter 
from ._seik import Seik

class CtrlRun(EnsFilter):
    def __init__(self, EnsSize, weights=None, forget=1.0, with_autotuning=False, autotuning_bounds=None):
        super().__init__(EnsSize, weights, forget, with_autotuning, autotuning_bounds)
        self.seik=Seik(EnsSize)
    
    def sampling(self, mean_and_base):
        return self.seik.sampling(mean_and_base)

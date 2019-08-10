import numpy as np
import pandas as pd
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.NumbaFuncs import get_bin_indices, weight_numba

def evaluate_pu(ev, source, nsig, var, corrs):
    mins = corrs["nTrueInt"].values.astype(np.float32)
    maxs = mins[:]+1
    mins[0] = -np.inf
    maxs[-1] = np.inf
    indices = get_bin_indices([getattr(ev, var)], [mins], [maxs], 1)[:,0]
    ev_corrs = corrs.iloc[indices]

    nominal = ev_corrs["corr"].values
    up = (ev_corrs["corr_up"].values/nominal - 1.)*(source=="pileup")
    down = (ev_corrs["corr_down"].values/nominal - 1.)*(source=="pileup")
    return weight_numba(nominal, nsig, up, down).astype(np.float32)

class WeightPileup(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        self.dfc = read_file(self.correction_file)
        event.register_function(
            event, "WeightPU", partial(
                evaluate_pu, var=self.variable, corrs=self.dfc,
            ),
        )

def read_file(path):
    return pd.read_csv(path, sep='\s+')[
        ["nTrueInt", "corr", "corr_down", "corr_up"]
    ]

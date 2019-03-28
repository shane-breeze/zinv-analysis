import numpy as np
import pandas as pd
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.NumbaFuncs import get_bin_indices, weight_numba

def evaluate_pu(var, corrs):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_pu'))
    def fevaluate_pu(ev, evidx, nsig, source, var_):
        mins = corrs["nTrueInt"].values.astype(np.float32)
        maxs = mins[:]+1
        maxs[-1] = np.inf
        indices = get_bin_indices([getattr(ev, var_)], [mins], [maxs], 1)[:,0]
        ev_corrs = corrs.iloc[indices]

        nominal = ev_corrs["corr"].values
        up = (ev_corrs["corr_up"].values/nominal - 1.)*(source=="pileup")
        down = (ev_corrs["corr_down"].values/nominal - 1.)*(source=="pileup")
        return weight_numba(nominal, nsig, up, down).astype(np.float32)

    def ret_func(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ["pileup"]:
            source, nsig = '', 0.
        return fevaluate_pu(ev, ev.iblock, nsig, source, var)

    return ret_func

class WeightPileup(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        self.dfc = read_file(self.correction_file)
        event.WeightPU = evaluate_pu(self.variable, self.dfc)

def read_file(path):
    return pd.read_csv(path, sep='\s+')[
        ["nTrueInt", "corr", "corr_down", "corr_up"]
    ]

import numpy as np
import pandas as pd

from numba import njit
from cachetools.func import lru_cache

from utils.NumbaFuncs import get_bin_indices, weight_numba

def evaluate_pu(var, corrs):
    @lru_cache(maxsize=32)
    def fevaluate_pu(ev, evidx, nsig, source, var_):
        vals = corrs["nTrueInt"].values
        indices = get_bin_indices(getattr(ev, var_), vals, vals+1)
        ev_corrs = corrs.iloc[indices]

        nominal = ev_corrs["corr"].values
        up = (ev_corrs["corr_up"].values/nominal - 1.)*(source=="pileup")
        down = (ev_corrs["corr_down"].values/nominal - 1.)*(source=="pileup")
        return weight_numba(nominal, nsig, up, down)

    return lambda ev: fevaluate_pu(ev, ev.iblock, ev.nsig, ev.source, var)

class WeightPileup(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        self.dfc = read_file(self.correction_file, overflow_bins=["nTrueInt"])
        event.WeightPU = evaluate_pu(self.variable, self.dfc)

def read_file(path, overflow_bins=[]):
    df = pd.read_table(path, sep='\s+')[["nTrueInt", "corr", "corr_down", "corr_up"]]
    df.loc[df["nTrueInt"]==df["nTrueInt"].max(), "nTrueInt"] = np.inf
    return df

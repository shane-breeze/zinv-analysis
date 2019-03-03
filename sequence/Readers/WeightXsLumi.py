import numpy as np
from numba import njit
from cachetools.func import lru_cache

from utils.NumbaFuncs import weight_numba

def evaluate_weight(sf):
    @lru_cache(maxsize=32)
    def fevaluate_weight(ev, parent, iblock, nsig, source, sf_):
        if parent in ['SingleTop', 'QCD']:
            return ev.genWeight * sf_
        else:
            try:
                lhe_weights = getattr(ev, 'LHE{}Weight'.format(source))-1.
            except AttributeError:
                lhe_weights = 0.
            return weight_numba(ev.genWeight*sf_, nsig, lhe_weights, lhe_weights)

    return lambda ev: fevaluate_weight(
        ev, ev.config.dataset.parent, ev.iblock, ev.nsig, ev.source, sf,
    )

class WeightXsLumi(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        dataset = event.config.dataset
        sumweights = sum([
            associates.sumweights
            for associates in dataset.associates
        ])
        sf = (dataset.xsection * dataset.lumi / sumweights)
        event.WeightXsLumi = evaluate_weight(sf)

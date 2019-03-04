import numpy as np
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial
from numba import njit

from utils.NumbaFuncs import weight_numba

def evaluate_xslumi_weight(sf):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_xslumi_weight'))
    def fevaluate_xslumi_weight(ev, parent, iblock, nsig, source, sf_):
        if parent in ['SingleTop', 'QCD']:
            return ev.genWeight * sf_
        else:
            try:
                lhe_weights = getattr(ev, 'LHE{}Weight'.format(source))-1.
            except AttributeError:
                lhe_weights = 0.
            return weight_numba(ev.genWeight*sf_, nsig, lhe_weights, lhe_weights)

    def ret_func(ev):
        source = ev.source
        if source not in ["Scale", "Pdf"]:
            source = ''
        return fevaluate_xslumi_weight(
            ev, ev.config.dataset.parent, ev.iblock, ev.nsig, source, sf,
        )

    return ret_func

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
        event.WeightXsLumi = evaluate_xslumi_weight(sf)

import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

def evaluate_xslumi_weight(sf):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_xslumi_weight'))
    def fevaluate_xslumi_weight(ev, iblock, sf_):
        return ev.genWeight*sf_

    return lambda ev: fevaluate_xslumi_weight(ev, ev.iblock, sf)

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

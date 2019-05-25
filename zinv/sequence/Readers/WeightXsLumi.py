from functools import partial

def evaluate_xslumi_weight(ev, sf, dataset):
    return ev.genWeight*sf

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

        event.register_function(
            event, "WeightXsLumi",
            partial(evaluate_xslumi_weight, sf=sf, dataset=dataset),
        )

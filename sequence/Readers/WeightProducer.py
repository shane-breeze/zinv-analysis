import numpy as np
import yaml

from cachetools.func import lru_cache

from utils.NumbaFuncs import prod_numba
from utils.Lambda import Lambda

def evaluate_weights(name, weights):
    @lru_cache(maxsize=32)
    def fevaluate_weights(ev, evidx, nsig, source, name_):
        return prod_numba(np.vstack([w(ev) for w in weights]).T)
    return lambda ev: fevaluate_weights(ev, ev.iblock, ev.nsig, ev.source, name)

class WeightProducer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        with open(self.weight_sequence_path, 'r') as f:
            input_dict = yaml.load(f)

        weights_dict = input_dict["weights"]
        event.variation_sources = input_dict["variations"]
        data_or_mc = "Data" if event.config.dataset.isdata else "MC"

        self.lambda_functions = {
            f: Lambda(f)
            for _, sdict in weights_dict.items()
            for _, flist in sdict.items()
            for f in flist
        }

        for dataset, subdict in weights_dict.items():
            weights = [self.lambda_functions[s] for s in subdict[data_or_mc]]
            setattr(event, "Weight_{}".format(dataset), evaluate_weights(
                dataset, weights,
            ))

    def end(self):
        self.lambda_functions = None

import numpy as np
import yaml
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.Lambda import Lambda

def evaluate_weights(name, weights):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_weights'))
    def fevaluate_weights(ev, evidx, nsig, source, name_):
        return reduce(operator.mul, weights)(ev)
    return lambda ev: fevaluate_weights(ev, ev.iblock, ev.nsig, ev.source, name)

class WeightProducer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        with open(self.weight_sequence_path, 'r') as f:
            input_dict = yaml.load(f)

        weights_dict = input_dict["weights"]
        wvars = input_dict["weight_variations"]
        avars = input_dict["attribute_variations"]
        if self.nuisances is not None:
            self.weight_variation_sources = [
                v for v in wvars if v in self.nuisances
            ]
            self.attribute_variation_sources = [
                v for v in avars if v in self.nuisances
            ]
        else:
            self.weight_variation_sources = wvars
            self.attribute_variation_sources = avars
        data_or_mc = "Data" if event.config.dataset.isdata else "MC"

        self.lambda_functions = {
            f: Lambda(f)
            for _, labeldicts in weights_dict.items()
            for _, regiondicts in labeldicts.items()
            for f in regiondicts["Data"]
        }
        self.lambda_functions.update({
            f: Lambda(f)
            for _, labeldicts in weights_dict.items()
            for _, regiondicts in labeldicts.items()
            for f in regiondicts["MC"]
        })

        for dataset, labeldicts in weights_dict.items():
            for label, regiondicts in labeldicts.items():
                weights = [self.lambda_functions[s] for s in regiondicts[data_or_mc]]
                weighter = evaluate_weights("_".join([dataset, label, data_or_mc]), weights)

                for region in regiondicts["Regions"]:
                    setattr(event, "Weight_{}_{}_{}".format(dataset, region, data_or_mc), weighter)

        event.weight_variation_sources = self.weight_variation_sources
        event.attribute_variation_sources = self.attribute_variation_sources

    def event(self, event):
        event.weight_variation_sources = self.weight_variation_sources
        event.attribute_variation_sources = self.attribute_variation_sources

    def end(self):
        self.lambda_functions = None

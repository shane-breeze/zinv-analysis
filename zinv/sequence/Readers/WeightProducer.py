import numpy as np
import yaml
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.Lambda import Lambda

def expand_pdf(nuisance_list):
    if "pdf" in nuisance_list:
        nuisance_list.remove("pdf")
        for i in range(1, 101):
            nuisance_list.append("pdf{}".format(i))
    return nuisance_list

class WeightProducer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        with open(self.weight_sequence_path, 'r') as f:
            input_dict = yaml.load(f)

        weights_dict = input_dict["weights"]
        wvars = input_dict["weight_variations"]
        wvars = expand_pdf(wvars)
        avars = input_dict["attribute_variations"]
        self.nuisances = None
        if self.nuisances is not None:
            self.nuisances = expand_pdf(self.nuisances)
            self.weight_variation_sources = [
                v for v in wvars if v in self.nuisances
            ]
            self.attribute_variation_sources = [
                v for v in avars if v in self.nuisances
            ]
        else:
            self.weight_variation_sources = wvars
            self.attribute_variation_sources = avars

        event.weight_variation_sources = self.weight_variation_sources
        event.attribute_variation_sources = self.attribute_variation_sources

    def event(self, event):
        event.weight_variation_sources = self.weight_variation_sources
        event.attribute_variation_sources = self.attribute_variation_sources

import yaml
import numpy as np
import awkward as awk
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial, reduce

from zinv.utils.Lambda import Lambda

def evaluate_skim(ev, source, nsig, objname, name, cutlist):
    starts = getattr(ev, objname).pt.starts
    stops = getattr(ev, objname).pt.stops

    return awk.JaggedArray(
        starts, stops,
        reduce(operator.add, cutlist)(ev, source, nsig).content,
    )

def evaluate_this_not_that(ev, source, nsig, this, that):
    this_attr = getattr(ev, this)(ev, source, nsig)
    that_attr = getattr(ev, that)(ev, source, nsig)
    return this_attr & (~that_attr)

class SkimCollections(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        with open(self.physics_object_selection_path, 'r') as f:
            selection_functions = yaml.load(f)

        self.lambda_functions = {
            f: Lambda(f)
            for _, v in selection_functions.items()
            for f in v["selections"]
        }

        for outcoll, subdict in selection_functions.items():
            incoll = subdict["original"]
            selections = [self.lambda_functions[s] for s in subdict["selections"]]
            name = "{}_{}Mask".format(incoll, outcoll)

            event.register_function(
                event, name, partial(
                    evaluate_skim, objname=incoll, name=name,
                    cutlist=selections,
                ),
            )

        for outcoll, subdict in selection_functions.items():
            if "Veto" not in outcoll:
                continue
            incoll = subdict["original"]
            name = "{}_{}Mask".format(incoll, outcoll)
            nosele_name = "{}_{}NoSelectionMask".format(incoll, outcoll)

            event.register_function(
                event, nosele_name, partial(
                    evaluate_this_not_that, this=name,
                    that=name.replace("Veto", "Selection"),
                ),
            )

    def end(self):
        self.lambda_functions = None

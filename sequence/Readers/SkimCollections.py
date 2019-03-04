import yaml
import numpy as np
import awkward as awk
import operator
from numba import njit

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from utils.NumbaFuncs import all_numba
from utils.Lambda import Lambda

def evaluate_skim(objname, name, cutlist):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_skim'))
    def fevaluate_skim(ev, evidx, nsig, source, name_, objname_):
        starts = getattr(ev, objname_).pt.starts
        stops = getattr(ev, objname_).pt.stops
        return awk.JaggedArray(
            starts, stops,
            all_numba(np.vstack([c(ev).content for c in cutlist]).T),
        )
    return lambda ev: fevaluate_skim(ev, ev.iblock, ev.nsig, ev.source, name, objname)

def evaluate_this_not_that(this, that):
    @njit
    def this_not_that_numba(this_, that_):
        return this_ & (~that_)

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_this_not_that'))
    def fevaluate_this_not_that(ev, evidx, nsig, source, this_, that_):
        this_attr = getattr(ev, this_)(ev)
        that_attr = getattr(ev, that_)(ev)
        return awk.JaggedArray(
            this_attr.starts, this_attr.stops,
            this_not_that_numba(this_attr.content, that_attr.content),
        )
    return lambda ev: fevaluate_this_not_that(ev, ev.iblock, ev.nsig, ev.source, this, that)

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
            setattr(event, name, evaluate_skim(incoll, name, selections))

        for outcoll, subdict in selection_functions.items():
            if "Veto" not in outcoll:
                continue
            incoll = subdict["original"]
            name = "{}_{}Mask".format(incoll, outcoll)
            nosele_name = "{}_{}NoSelectionMask".format(incoll, outcoll)

            setattr(event, nosele_name, evaluate_this_not_that(
                name, name.replace("Veto", "Selection"),
            ))

    def end(self):
        self.lambda_functions = None

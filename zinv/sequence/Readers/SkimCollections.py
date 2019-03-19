import yaml
import numpy as np
import awkward as awk
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.Lambda import Lambda

def evaluate_skim(objname, name, cutlist):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_skim'))
    def fevaluate_skim(ev, evidx, nsig, source, name_, objname_):
        starts = getattr(ev, objname_).pt.starts
        stops = getattr(ev, objname_).pt.stops
        return awk.JaggedArray(
            starts, stops,
            reduce(operator.add, cutlist)(ev).content,
        )

    def return_evaluate_skim(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fevaluate_skim(ev, ev.iblock, nsig, source, name, objname)

    return return_evaluate_skim

def evaluate_this_not_that(this, that):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_this_not_that'))
    def fevaluate_this_not_that(ev, evidx, nsig, source, this_, that_):
        this_attr = getattr(ev, this_)(ev)
        that_attr = getattr(ev, that_)(ev)
        return this_attr & (~that_attr)

    def return_evaluate_this_not_that(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ev.attribute_variation_sources:
            source, nsig = '', 0.
        return fevaluate_this_not_that(ev, ev.iblock, nsig, source, this, that)

    return return_evaluate_this_not_that

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

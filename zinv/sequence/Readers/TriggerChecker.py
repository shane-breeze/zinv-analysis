import re
import yaml
import numpy as np
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial, reduce

def evaluate_triggers(ev, triggers):
    return reduce(operator.or_, [
        getattr(ev, trigger)
        for trigger in triggers
        if ev.hasbranch(trigger)
    ])

class TriggerChecker(object):
    regex = re.compile("^(?P<dataset>[a-zA-Z0-9]*)_Run2016(?P<run_letter>[a-zA-Z])_v(?P<version>[0-9])$")
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        with open(self.trigger_selection_path, 'r') as f:
            input_dict = yaml.load(f)

        isdata = event.config.dataset.isdata
        data_or_mc = "Data" if isdata else "MC"
        trigger_dict = input_dict[data_or_mc]
        if isdata:
            match = self.regex.search(event.config.dataset.name)
            if match:
                dataset = match.group("dataset")
                run = match.group("run_letter")+match.group("version")

            trigger_dict["MET"] = trigger_dict["MET"][run]
        else:
            dataset = "MET"

        if not isdata:
            for dataset in trigger_dict.keys():
                event.register_function(
                    event,
                    "Is{}Triggered".format(dataset),
                    lambda ev: np.ones(ev.size, dtype=float),
                )
        else:
            for dataset, trigger_list in trigger_dict.items():
                event.register_function(
                    event,
                    "Is{}Triggered".format(dataset),
                    partial(evaluate_triggers, triggers=trigger_list),
                )

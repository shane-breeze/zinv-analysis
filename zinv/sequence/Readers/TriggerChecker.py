import re
import yaml
import numpy as np
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

def evaluate_triggers(triggers):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_triggers'))
    def fevaluate_triggers(ev, evidx, triggers_list):
        return reduce(operator.or_, [
            getattr(ev, trigger)
            for trigger in triggers_list
            if ev.hasbranch(trigger)
        ])
    return lambda ev: fevaluate_triggers(ev, ev.iblock, tuple(triggers))

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
                setattr(
                    event,
                    "Is{}Triggered".format(dataset),
                    lambda ev: np.ones(ev.size, dtype=float),
                )
        else:
            for dataset, trigger_list in trigger_dict.items():
                setattr(
                    event,
                    "Is{}Triggered".format(dataset),
                    evaluate_triggers(trigger_list),
                )
        event.IsTriggered = getattr(event, "Is{}Triggered".format(dataset))

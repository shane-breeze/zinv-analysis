import numpy as np
import yaml
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial, reduce

from zinv.utils.Lambda import Lambda

def evaluate_selection(ev, source, nsig, name, cutlist):
    return reduce(
        operator.add, cutlist,
        Lambda("ev, source, nsig: np.ones(ev.size, dtype=np.bool8)"),
    )(ev, source, nsig)

class SelectionProducer(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        with open(self.event_selection_path, 'r') as f:
            event_selection = yaml.load(f)

        selections = event_selection["selections"]

        grouped_selections = event_selection["grouped_selections"]
        expanded_grouped_selections = {}
        for label, selectionlist in grouped_selections.items():
            if label not in expanded_grouped_selections:
                expanded_grouped_selections[label] = []
            expanded_grouped_selections[label].extend([
                (s, selections[s]) for s in selectionlist
            ])

        cutflows = event_selection["cutflows"]
        expanded_cutflows = {}
        for cutflow, datamc_dict in cutflows.items():
            if cutflow not in expanded_cutflows:
                expanded_cutflows[cutflow] = {}

            for data_or_mc, labellist in datamc_dict.items():
                if data_or_mc not in expanded_cutflows[cutflow]:
                    expanded_cutflows[cutflow][data_or_mc] = []

                for label in labellist:
                    expanded_cutflows[cutflow][data_or_mc].extend(
                        expanded_grouped_selections[label]
                    )

        self.selections = expanded_cutflows

        # Create N-1 cutflows
        additional_selections = {}
        for cutflow, datamc_selections in self.selections.items():
            for data_or_mc, selection in datamc_selections.items():
                for subselection in selection:
                    new_selection = selection[:]
                    new_selection.remove(subselection)
                    newcutflow = "{}_remove_{}".format(cutflow, subselection[0])

                    if newcutflow not in additional_selections:
                        additional_selections[newcutflow] = {}
                        additional_selections[newcutflow][data_or_mc] = new_selection

        self.selections.update(additional_selections)

        # Add cutflows to the event
        data_or_mc = "Data" if event.config.dataset.isdata else "MC"
        for cutflow, datamc_selections in self.selections.items():
            if data_or_mc in datamc_selections:
                event.register_function(
                    event, "Cutflow_"+cutflow,
                    partial(
                        evaluate_selection, name=cutflow,
                        cutlist=[
                            Lambda(cut[1])
                            for cut in datamc_selections[data_or_mc]
                        ],
                    ),
                )

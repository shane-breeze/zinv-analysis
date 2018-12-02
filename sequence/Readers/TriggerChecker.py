import re
import numpy as np

trigger_selection = {
    "MET": {
        "B1": ['HLT_PFMETNoMu90_PFMHTNoMu90_IDTight',
               'HLT_PFMETNoMu100_PFMHTNoMu100_IDTight',
               'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight',
               'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
               'HLT_PFMET170_NotCleaned',
               'HLT_PFMET170_BeamHaloCleaned',
               'HLT_PFMET170_HBHECleaned',
               'HLT_PFMET170_HBHE_BeamHaloCleaned'],
        "B2": ['HLT_PFMETNoMu100_PFMHTNoMu100_IDTight',
               'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight',
               'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
               'HLT_PFMET170_NotCleaned',
               'HLT_PFMET170_BeamHaloCleaned',
               'HLT_PFMET170_HBHECleaned',
               'HLT_PFMET170_HBHE_BeamHaloCleaned'],
        "C1": ['HLT_PFMETNoMu100_PFMHTNoMu100_IDTight',
               'HLT_PFMETNoMu110_PFMHTNoMu110_IDTight',
               'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
               'HLT_PFMET170_HBHECleaned',
               'HLT_PFMET170_HBHE_BeamHaloCleaned'],
        "D1": ['HLT_PFMETNoMu110_PFMHTNoMu110_IDTight',
               'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
               'HLT_PFMET170_HBHECleaned',
               'HLT_PFMET170_HBHE_BeamHaloCleaned'],
        "E1": ['HLT_PFMETNoMu110_PFMHTNoMu110_IDTight',
               'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
               'HLT_PFMET170_HBHECleaned',
               'HLT_PFMET170_HBHE_BeamHaloCleaned'],
        "F1": ['HLT_PFMETNoMu110_PFMHTNoMu110_IDTight',
               'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
               'HLT_PFMET170_HBHECleaned',
               'HLT_PFMET170_HBHE_BeamHaloCleaned'],
        "G1": ['HLT_PFMETNoMu110_PFMHTNoMu110_IDTight',
               'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
               'HLT_PFMET170_HBHECleaned',
               'HLT_PFMET170_HBHE_BeamHaloCleaned'],
        "H2": ['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
               'HLT_PFMET170_HBHECleaned',
               'HLT_PFMET170_HBHE_BeamHaloCleaned'],
        "H3": ['HLT_PFMETNoMu110_PFMHTNoMu110_IDTight',
               'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
               'HLT_PFMET170_HBHECleaned',
               'HLT_PFMET170_HBHE_BeamHaloCleaned'],
    },
    "SingleMuon": ["HLT_IsoMu24", "HLT_IsoTkMu24"],
    "SingleElectron": ["HLT_Ele27_WPTight_Gsf"]
}

trigger_selection_mc = {
    "MET": ['HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
            'HLT_PFMET170_HBHECleaned',
            'HLT_PFMET170_HBHE_BeamHaloCleaned'],
    "SingleMuon": ["HLT_IsoMu24", "HLT_IsoTkMu24"],
    "SingleElectron": ["HLT_Ele27_WPTight_Gsf"]
}

class TriggerChecker(object):
    regex = re.compile("^(?P<dataset>[a-zA-Z0-9]*)_Run2016(?P<run_letter>[a-zA-Z])_v(?P<version>[0-9])$")
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        self.trigger_dict = trigger_selection
        self.isdata = event.config.dataset.isdata
        if not self.isdata:
            self.trigger_dict = trigger_selection_mc
            return
        match = self.regex.search(event.config.dataset.name)
        if match:
            self.dataset = match.group("dataset")
            self.run = match.group("run_letter")+match.group("version")

        self.trigger_dict["MET"] = self.trigger_dict["MET"][self.run]

    def event(self, event):
        # MC
        if not self.isdata:
            for dataset in self.trigger_dict.keys():
                setattr(event, "Is{}Triggered".format(dataset), np.ones(event.size, dtype=bool))
                event.IsTriggered = np.ones(event.size, dtype=bool)
            return

        # Data
        for dataset, trigger_list in self.trigger_dict.items():
            setattr(
                event,
                "Is{}Triggered".format(dataset),
                reduce(
                    lambda x,y: x|y,
                    [getattr(event, trigger)
                     for trigger in trigger_list
                     if event.hasbranch(trigger)],
                ),
            )
        event.IsTriggered = getattr(event, "Is{}Triggered".format(self.dataset))

import logging
import numpy as np

from collections import OrderedDict as odict
from utils.Lambda import Lambda

class SelectionProducer(object):
    attr_variation_conv = odict([
        ("ev.nBJetSelectionMedium",  "ev.nBJetSelectionMedium{}"),
        ("ev.MinDPhiJ1234METnoX",    "ev.MinDPhiJ1234METnoX{}"),
        ("ev.MET_pt",                "ev.MET_pt{}"),
        ("ev.MET_phi",               "ev.MET_phi{}"),
        ("ev.MET_dCaloMET",          "ev.MET_dCaloMET{}"),
        ("ev.METnoX_pt",             "ev.METnoX_pt{}"),
        ("ev.METnoX_phi",            "ev.METnoX_phi{}"),
        ("ev.Jet_pt",                "ev.Jet_pt{}"),
        ("ev.Jet_mass",              "ev.Jet_mass{}"),
        ("ev.MET.pt",                "ev.MET.pt{}"),
        ("ev.MET.phi",               "ev.MET.phi{}"),
        ("ev.MET.dCaloMET",          "ev.MET.dCaloMET{}"),
        ("ev.METnoX.pt",             "ev.METnoX.pt{}"),
        ("ev.METnoX.phi",            "ev.METnoX.phi{}"),
        ("ev.Jet.pt",                "ev.Jet.pt{}"),
        ("ev.Jet.mass",              "ev.Jet.mass{}"),
        # Collections - careful here
        ("ev.JetVeto.pt",            "ev.JetVeto.pt{}"),
        ("ev.JetVeto.mass",          "ev.JetVeto.mass{}"),
        ("ev.JetVeto",               "ev.JetVeto{}"),
        ("ev.JetSelection.pt",       "ev.JetSelection.pt{}"),
        ("ev.JetSelection.mass",     "ev.JetSelection.mass{}"),
        ("ev.JetSelection",          "ev.JetSelection{}"),
        ("ev.HMiss.pt",              "ev.HMiss.pt{}"),
        ("ev.HMiss.eta",             "ev.HMiss.eta{}"),
        ("ev.HMiss.phi",             "ev.HMiss.phi{}"),
    ])
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        self.isdata = event.config.dataset.isdata
        es = self.event_selection

        baseline = es.data_selection if self.isdata else es.mc_selection
        self.selections = {
            "None": [],
            "Monojet": baseline + es.baseline_selection + es.monojet_selection,
            "Monojet_noMETTrigger": [(n, s)
                                     for (n, s) in baseline \
                                     + es.baseline_selection \
                                     + es.monojet_selection
                                     if n not in ["met_selection", "blind_mask"]],
            "Monojet_METTrigger": [(n, s)
                                   for (n, s) in baseline \
                                   + es.met_trigger_selection \
                                   + es.baseline_selection \
                                   + es.monojet_selection
                                   if n not in ["met_selection", "blind_mask"]],
            "Monojet_unblind": [(n, s)
                                for (n, s) in baseline \
                                + es.baseline_selection \
                                + es.monojet_selection
                                if n not in ["met_selection", "blind_mask", "muon_selection_fmt_0"]],
            "MonojetQCD": baseline + es.baseline_selection + es.monojetqcd_selection,
            "Monojet_remove_muon_selection_fmt_0": [(n, s)
                                                    for (n, s) in baseline \
                                                    + es.baseline_selection \
                                                    + es.monojet_selection
                                                    if n not in ["muon_selection_fmt_0"]],
            "SingleMuon": baseline + es.baseline_selection + es.singlemuon_selection,
            "SingleMuon_noMETTrigger": [(n, s)
                                        for (n, s) in baseline\
                                        + es.baseline_selection\
                                        + es.singlemuon_selection
                                        if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "SingleMuon_METTrigger": [(n, s)
                                      for (n, s) in baseline\
                                      + es.met_trigger_selection\
                                      + es.baseline_selection\
                                      + es.singlemuon_selection
                                      if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "SingleMuonQCD": baseline + es.baseline_selection + es.singlemuonqcd_selection,
            "SingleMuonPlus": baseline + es.baseline_selection + es.singlemuon_selection + es.singlemuonplus_selection,
            "SingleMuonMinus": baseline + es.baseline_selection + es.singlemuon_selection + es.singlemuonminus_selection,
            "DoubleMuon": baseline + es.baseline_selection + es.doublemuon_selection,
            "DoubleMuon_noMETTrigger": [(n, s)
                                        for (n, s) in baseline\
                                        + es.baseline_selection\
                                        + es.doublemuon_selection
                                        if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "DoubleMuon_METTrigger": [(n, s)
                                      for (n, s) in baseline\
                                      + es.met_trigger_selection\
                                      + es.baseline_selection\
                                      + es.doublemuon_selection
                                      if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "DoubleMuon_unblind": [(n, s)
                                   for (n, s) in baseline \
                                   + es.baseline_selection \
                                   + es.doublemuon_selection
                                   if n not in ["met_selection", "blind_mask"]],
            "TripleMuon_noMETTrigger": [(n, s)
                                        for (n, s) in baseline\
                                        + es.baseline_selection\
                                        + es.triplemuon_selection
                                        if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "TripleMuon_METTrigger": [(n, s)
                                      for (n, s) in baseline\
                                      + es.met_trigger_selection\
                                      + es.baseline_selection\
                                      + es.triplemuon_selection
                                      if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "QuadMuon_noMETTrigger": [(n, s)
                                      for (n, s) in baseline\
                                      + es.baseline_selection\
                                      + es.quadmuon_selection
                                      if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "QuadMuon_METTrigger": [(n, s)
                                    for (n, s) in baseline\
                                    + es.met_trigger_selection\
                                    + es.baseline_selection\
                                    + es.quadmuon_selection
                                    if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "SingleElectron": baseline + es.baseline_selection + es.singleelectron_selection,
            "SingleElectron_noMETTrigger": [(n, s)
                                            for (n, s) in baseline\
                                            + es.baseline_selection\
                                            + es.singleelectron_selection
                                            if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "SingleElectron_METTrigger": [(n, s)
                                          for (n, s) in baseline\
                                          + es.met_trigger_selection\
                                          + es.baseline_selection\
                                          + es.singleelectron_selection
                                          if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "SingleElectronQCD": baseline + es.baseline_selection + es.singleelectronqcd_selection,
            "SingleElectronPlus": baseline + es.baseline_selection + es.singleelectron_selection + es.singleelectronplus_selection,
            "SingleElectronMinus": baseline + es.baseline_selection + es.singleelectron_selection + es.singleelectronminus_selection,
            "DoubleElectron": baseline + es.baseline_selection + es.doubleelectron_selection,
            "DoubleElectron_noMETTrigger": [(n, s)
                                            for (n, s) in baseline\
                                            + es.baseline_selection\
                                            + es.doubleelectron_selection
                                            if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "DoubleElectron_METTrigger": [(n, s)
                                          for (n, s) in baseline\
                                          + es.met_trigger_selection\
                                          + es.baseline_selection\
                                          + es.doubleelectron_selection
                                          if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "DoubleElectron_unblind": [(n, s)
                                       for (n, s) in baseline \
                                       + es.baseline_selection \
                                       + es.doubleelectron_selection
                                       if n not in ["met_selection", "blind_mask"]],
            "DoubleElectronAlt": baseline + es.doubleelectron_alt_selection,
            "TripleElectron_noMETTrigger": [(n, s)
                                            for (n, s) in baseline\
                                            + es.baseline_selection\
                                            + es.tripleelectron_selection
                                            if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "TripleElectron_METTrigger": [(n, s)
                                          for (n, s) in baseline\
                                          + es.met_trigger_selection\
                                          + es.baseline_selection\
                                          + es.tripleelectron_selection
                                          if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "QuadElectron_noMETTrigger": [(n, s)
                                          for (n, s) in baseline\
                                          + es.baseline_selection\
                                          + es.quadelectron_selection
                                          if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "QuadElectron_METTrigger": [(n, s)
                                        for (n, s) in baseline\
                                        + es.met_trigger_selection\
                                        + es.baseline_selection\
                                        + es.quadelectron_selection
                                        if n not in ["met_selection", "blind_mask", "mtw_selection", "mll_selection"]],
            "SingleTau": baseline + es.baseline_selection + es.singletau_selection,
            "SingleTauQCD": baseline + es.baseline_selection + es.singletauqcd_selection,
            "DoubleTau": baseline + es.baseline_selection + es.doubletau_selection,
        }

        # Create N-1 cutflows
        additional_selections = {}
        for cutflow, selection in self.selections.items():
            for subselection in selection:
                if subselection[0] == "blind_mask":
                    continue
                new_selection = selection[:]
                new_selection.remove(subselection)
                newcutflow = "{}_remove_{}".format(cutflow, subselection[0])
                additional_selections[newcutflow] = new_selection

        # Create variation cutflows
        for cutflow, selection in self.selections.items():
            for variation in event.variations:
                if variation == "":
                    continue
                new_selection = selection[:]
                for attr, new_attr in self.attr_variation_conv.items():
                    new_selection = [
                        (subselection[0],
                         subselection[1].replace(attr, new_attr.format(variation))) \
                        if attr in subselection[1] and not self.isdata \
                        else subselection
                        for subselection in new_selection
                    ]
                additional_selections[cutflow+variation] = new_selection

        self.selections.update(additional_selections)
        self.selections_lambda = {cutflow: [Lambda(cut) for name, cut in selection]
                                  for cutflow, selection in self.selections.items()}

    def event(self, event):
        for cutflow, selection in self.selections_lambda.items():
            cuts = np.ones(event.size, dtype=bool)
            if len(selection) > 0:
                for cut in selection:
                    cuts = cuts & cut(event)
            setattr(event, "Cutflow_{}".format(cutflow), cuts)

    def end(self):
        self.selections_lambda = {}

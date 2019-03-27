import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.sequence.Readers import WeightObjects

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.attribute_variation_sources = []
        self.cache = {}

@pytest.fixture()
def path():
    toppath = os.path.abspath(os.environ["TOPDIR"])
    datapath = os.path.join(toppath, "zinv/data")
    return datapath

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module_electrons(path):
    return WeightObjects(
        correctors = [
            {
                "name": "eleIdIsoTight",
                "collection": "Electron",
                "binning_variables": ("ev: ev.Electron.eta", "ev: ev.Electron_ptShift(ev)"),
                "weighted_paths": [(1, path+"/electrons/electron_idiso_tight.txt")],
                "add_syst": "ev: awk.JaggedArray.zeros_like(ev.Electron.eta)",
                "nuisances": ["eleIdIsoTight", "eleEnergyScale"],
            }, {
                "name": "eleIdIsoVeto",
                "collection": "Electron",
                "binning_variables": ("ev: ev.Electron.eta", "ev: ev.Electron_ptShift(ev)"),
                "weighted_paths": [(1, path+"/electrons/electron_idiso_veto.txt")],
                "add_syst": "ev: awk.JaggedArray.zeros_like(ev.Electron.eta)",
                "nuisances": ["eleIdIsoVeto", "eleEnergyScale"],
            }, {
                "name": "eleReco",
                "collection": "Electron",
                "binning_variables": ("ev: ev.Electron.eta", "ev: ev.Electron_ptShift(ev)"),
                "weighted_paths": [(1, path+"/electrons/electron_reconstruction.txt")],
                "add_syst": "ev: 0.01*((ev.Electron_ptShift(ev)<20) | (ev.Electron_ptShift(ev)>80))",
                "nuisances": ["eleReco", "eleEnergyScale"],
            }, {
                "name": "eleTrig",
                "collection": "Electron",
                "binning_variables": ("ev: ev.Electron_ptShift(ev)", "ev: np.abs(ev.Electron.eta)"),
                "weighted_paths": [(1, path+"/electrons/electron_trigger_v2.txt")],
                "add_syst": "ev: awk.JaggedArray.zeros_like(ev.Electron.eta)",
                "nuisances": ["eleTrig", "eleEnergyScale"],
            },
        ],
    )

@pytest.fixture()
def module_muons(path):
    return WeightObjects(
        correctors = [
            {
                "name": "muonIdTight",
                "collection": "Muon",
                "binning_variables": ("ev: np.abs(ev.Muon.eta)", "ev: ev.Muon_ptShift(ev)"),
                "weighted_paths": [(19.7, path+"/muons/muon_id_loose_runBCDEF.txt"),
                                   (16.2, path+"/muons/muon_id_loose_runGH.txt")],
                "add_syst": "ev: 0.01*awk.JaggedArray.ones_like(ev.Muon.eta)",
                "nuisances": ["muonIdTight", "muonPtScale"],
            }, {
                "name": "muonIdLoose",
                "collection": "Muon",
                "binning_variables": ("ev: np.abs(ev.Muon.eta)", "ev: ev.Muon_ptShift(ev)"),
                "weighted_paths": [(19.7, path+"/muons/muon_id_loose_runBCDEF.txt"),
                                   (16.2, path+"/muons/muon_id_loose_runGH.txt")],
                "add_syst": "ev: 0.01*awk.JaggedArray.ones_like(ev.Muon.eta)",
                "nuisances": ["muonIdLoose", "muonPtScale"],
            }, {
                "name": "muonIsoTight",
                "collection": "Muon",
                "binning_variables": ("ev: np.abs(ev.Muon.eta)", "ev: ev.Muon_ptShift(ev)"),
                "weighted_paths": [(19.7, path+"/muons/muon_iso_tight_tightID_runBCDEF.txt"),
                                   (16.2, path+"/muons/muon_iso_tight_tightID_runGH.txt")],
                "add_syst": "ev: 0.005*awk.JaggedArray.ones_like(ev.Muon.eta)",
                "nuisances": ["muonIsoTight", "muonPtScale"],
            }, {
                "name": "muonIsoLoose",
                "collection": "Muon",
                "binning_variables": ("ev: np.abs(ev.Muon.eta)", "ev: ev.Muon_ptShift(ev)"),
                "weighted_paths": [(19.7, path+"/muons/muon_iso_loose_looseID_runBCDEF.txt"),
                                   (16.2, path+"/muons/muon_iso_loose_looseID_runGH.txt")],
                "add_syst": "ev: 0.005*awk.JaggedArray.ones_like(ev.Muon.eta)",
                "nuisances": ["muonIsoLoose", "muonPtScale"],
            }, {
                "name": "muonTrig",
                "collection": "Muon",
                "binning_variables": ("ev: np.abs(ev.Muon.eta)", "ev: ev.Muon_ptShift(ev)"),
                "weighted_paths": [(19.7, path + "/muons/muon_trigger_IsoMu24_OR_IsoTkMu24_runBCDEF.txt"),
                                   (16.2, path + "/muons/muon_trigger_IsoMu24_OR_IsoTkMu24_runGH.txt")],
                "add_syst": "ev: 0.005*awk.JaggedArray.ones_like(ev.Muon.eta)",
                "nuisances": ["muonTrig", "muonPtScale"],
            },
        ],
    )

@pytest.fixture()
def module_taus(path):
    return WeightObjects(
        correctors = [
            {
                "name": "tauIdTight",
                "collection": "Tau",
                "binning_variables": ("ev: ev.Tau_ptShift(ev)",),
                "weighted_paths": [(1, path+"/taus/tau_id_tight.txt")],
                "add_syst": "ev: 0.05*awk.JaggedArray.ones_like(ev.Tau.eta)",
                "nuisances": ["tauIdTight", "tauEnergyScale"],
            },
        ],
    )

@pytest.fixture()
def module_photons(path):
    return WeightObjects(
        correctors = [
            {
                "name": "photonIdLoose",
                "collection": "Photon",
                "binning_variables": ("ev: ev.Photon.eta", "ev: ev.Photon_ptShift(ev)"),
                "weighted_paths": [(1, path+"/photons/photon_cutbasedid_loose.txt")],
                "add_syst": "ev: awk.JaggedArray.zeros_like(ev.Photon.eta)",
                "nuisances": ["photonIdLoose", "photonEnergyScale"],
            }, {
                "name": "photonPixelSeedVeto",
                "collection": "Photon",
                "binning_variables": ("ev: ev.Photon.r9", "ev: np.abs(ev.Photon.eta)", "ev: ev.Photon_ptShift(ev)"),
                "weighted_paths": [(1, path+"/photons/photon_pixelseedveto.txt")],
                "add_syst": "ev: awk.JaggedArray.zeros_like(ev.Photon.eta)",
                "nuisances": ["photonPixelSeedVeto", "photonEnergyScale"],
            },
        ],
    )

#def test_weightobjects_init(module_electrons):


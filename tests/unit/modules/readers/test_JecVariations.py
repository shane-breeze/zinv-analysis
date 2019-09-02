import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.modules.readers import JecVariations

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.nsig = 0
        self.source = ''
        self.cache = {}
        self.attribute_variation_sources = [
            "jesTotal", "jerSF", "unclust", "jesAbsoluteStat",
        ]
        self.config = mock.MagicMock()

        self.Jet = DummyColl()
        self.GenJet = DummyColl()
        self.MET = DummyColl()
        self.PuppiMET = DummyColl()

    def register_function(self, event, name, function):
        self.__dict__[name] = function

    def hasbranch(self, branch):
        return hasattr(self, branch)

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return JecVariations(
        jes_unc_file = "http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/jecs/Summer16_23Sep2016V4_MC_UncertaintySources_AK4PFchs.csv",
        jer_sf_file = "http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/jecs/Spring16_25nsV10a_MC_SF_AK4PFchs.csv",
        jer_file = "http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/jecs/Spring16_25nsV10_MC_PtResolution_AK4PFchs.csv",
        apply_jer_corrections = True,
        jes_regex = "jes(?P<source>.*)",
        unclust_threshold = 15.,
        maxdr_jets_with_genjets = 0.2,
        ndpt_jets_with_genjets = 3.,
    )

def test_jec_variations_begin(module, event):
    module.begin(event)
    print(module.jes_sources)
    assert all(t in module.jes_sources for t in ["Total", "AbsoluteStat"])
    assert all(t in event.JesSources for t in ["jesTotal", "jesAbsoluteStat"])

@pytest.fixture()
def event_module_run(module, event):
    def norm(*args, **kwargs):
        return 0.5
    np.random.normal = mock.Mock(side_effect=norm)
    inputs = {
        "jpt":       [[], [16.],   [60., 70.]],
        "jeta":      [[], [-3.1],  [0.5, 3.1]],
        "jphi":      [[], [0.1],   [0.3, 0.5]],
        "gjpt":      [[], [17.],   [65., 75.]],
        "gjeta":     [[], [-3.11], [1.0, 3.09]],
        "gjphi":     [[], [0.1],   [0.3, 0.5]],
        "rho":       [0.,   20.,  50.],
        "met":       [200., 220., 240.],
        "mephi":     [0.9,  0.1,  -0.7],
        "met_sumet": [400., 600., 800.],
    }
    outputs = {
        "jpt":         [[], [18.624],            [76.29538585,    81.48]],
        "jptres":      [[], [0.456227396877664], [0.13144913,     0.179063214239771]],
        "jjersf":      [[], [1.164],             [1.271589764,    1.164]],
        "jjersfdown":  [[], [-0.05369415808],    [-0.06170861319, -0.05369415808]],
        "jjersfup":    [[], [0.05369415808],     [0.05007691581,  0.05369415808]],
        "jjestotdown": [[], [-0.07725],          [-0.0144,        -0.05163]],
        "jjestotup":   [[], [0.07725],           [0.0144,         0.05163]],
        "met":         [200., 217.376, 228.3443658],
        "mephi":       [0.9,  0.1,     -0.8071129825],
    }
    event.config.dataset.idx = 2
    event.size = 3

    jet_pt = awk.JaggedArray.fromiter(inputs["jpt"]).astype(np.float32)
    jet_eta = awk.JaggedArray.fromiter(inputs["jeta"]).astype(np.float32)
    jet_phi = awk.JaggedArray.fromiter(inputs["jphi"]).astype(np.float32)

    genjet_pt = awk.JaggedArray.fromiter(inputs["gjpt"]).astype(np.float32)
    genjet_eta = awk.JaggedArray.fromiter(inputs["gjeta"]).astype(np.float32)
    genjet_phi = awk.JaggedArray.fromiter(inputs["gjphi"]).astype(np.float32)

    rho = np.array(inputs["rho"], dtype=np.float32)
    met_pt = np.array(inputs["met"], dtype=np.float32)
    met_phi = np.array(inputs["mephi"], dtype=np.float32)
    met_sumet = np.array(inputs["met_sumet"], dtype=np.float32)

    event.Jet_pt = jet_pt
    event.Jet.pt = jet_pt
    event.Jet_ptJESOnly = jet_pt
    event.Jet.ptJESOnly = jet_pt
    event.Jet_eta = jet_eta
    event.Jet.eta = jet_eta
    event.Jet_phi = jet_phi
    event.Jet.phi = jet_phi

    event.GenJet_pt = genjet_pt
    event.GenJet.pt = genjet_pt
    event.GenJet_eta = genjet_eta
    event.GenJet.eta = genjet_eta
    event.GenJet_phi = genjet_phi
    event.GenJet.phi = genjet_phi

    event.fixedGridRhoFastjetAll = rho
    event.MET_pt = met_pt
    event.MET.pt = met_pt
    event.MET_ptJESOnly = met_pt
    event.MET.ptJESOnly = met_pt
    event.MET_phi = met_phi
    event.MET.phi = met_phi
    event.MET_phiJESOnly = met_phi
    event.MET.phiJESOnly = met_phi

    event.MET_sumEt = met_sumet
    event.MET.sumEt = met_sumet
    event.MET_sumEtJESOnly = met_sumet
    event.MET.sumEtJESOnly = met_sumet

    event.PuppiMET_pt = met_pt
    event.PuppiMET.pt = met_pt
    event.PuppiMET_ptJESOnly = met_pt
    event.PuppiMET.ptJESOnly = met_pt
    event.PuppiMET_phi = met_phi
    event.PuppiMET.phi = met_phi
    event.PuppiMET_phiJESOnly = met_phi
    event.PuppiMET.phiJESOnly = met_phi

    event.PuppiMET_sumEt = met_sumet
    event.PuppiMET.sumEt = met_sumet
    event.PuppiMET_sumEtJESOnly = met_sumet
    event.PuppiMET.sumEtJESOnly = met_sumet

    module.begin(event)
    #module.event(event)
    event.outputs = outputs
    return event, module

def test_jec_variations_ptres(event_module_run):
    event = event_module_run[0]
    outputs = event.outputs

    assert np.allclose(
        event.Jet_ptResolution(event).content,
        awk.JaggedArray.fromiter(outputs["jptres"]).astype(np.float32).content,
        rtol=1e-6, equal_nan=True,
    )

def test_jec_variations_jersfdown(event_module_run):
    event = event_module_run[0]
    outputs = event.outputs
    assert np.allclose(
        event.Jet_jerSF(event, "jerSF", -1.).content,
        awk.JaggedArray.fromiter(outputs["jjersfdown"]).astype(np.float32).content,
        rtol=1e-5, equal_nan=True,
    )

def test_jec_variations_jersfup(event_module_run):
    event = event_module_run[0]
    outputs = event.outputs
    assert np.allclose(
        event.Jet_jerSF(event, "jerSF", 1.).content,
        awk.JaggedArray.fromiter(outputs["jjersfup"]).astype(np.float32).content,
        rtol=1e-5, equal_nan=True,
    )

def test_jec_variations_newjpt(event_module_run):
    event, module = event_module_run
    module.event(event)
    outputs = event.outputs
    assert np.allclose(
        event.Jet_pt.content,
        awk.JaggedArray.fromiter(outputs["jpt"]).astype(np.float32).content,
        rtol=1e-6, equal_nan=True,
    )

def test_jec_variations_newmet(event_module_run):
    event, module = event_module_run
    module.event(event)
    outputs = event.outputs
    assert np.allclose(
        event.MET_pt,
        np.array(outputs["met"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

def test_jec_variations_newmephi(event_module_run):
    event, module = event_module_run
    module.event(event)
    outputs = event.outputs
    assert np.allclose(
        event.MET_phi,
        np.array(outputs["mephi"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

def test_jec_variations_jestotup(event_module_run):
    event, module = event_module_run
    module.event(event)
    outputs = event.outputs
    assert np.allclose(
        event.Jet_jesSF(event, "jesTotal", 1.).content,
        awk.JaggedArray.fromiter(outputs["jjestotup"]).astype(np.float32).content,
        rtol=1e-6, equal_nan=True,
    )

def test_jec_variations_jestotdown(event_module_run):
    event, module = event_module_run
    module.event(event)
    outputs = event.outputs
    assert np.allclose(
        event.Jet_jesSF(event, "jesTotal", -1.).content,
        awk.JaggedArray.fromiter(outputs["jjestotdown"]).astype(np.float32).content,
        rtol=1e-6, equal_nan=True,
    )

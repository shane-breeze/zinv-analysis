import pytest
import mock
import os
import numpy as np
import awkward as awk

from sequence.Readers import JecVariations

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

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    toppath = os.path.abspath(os.environ["TOPDIR"])
    datapath = os.path.join(toppath, "data")
    return JecVariations(
        jes_unc_file = datapath + "/jecs/Summer16_23Sep2016V4_MC_UncertaintySources_AK4PFchs.txt",
        jer_sf_file = datapath + "/jecs/Spring16_25nsV10a_MC_SF_AK4PFchs.txt",
        jer_file = datapath + "/jecs/Spring16_25nsV10_MC_PtResolution_AK4PFchs.txt",
        apply_jer_corrections = True,
        jes_regex = "jes(?P<source>.*)",
        unclust_threshold = 15.,
        maxdr_jets_with_genjets = 0.2,
        ndpt_jets_with_genjets = 3.,
    )

def test_jec_variations_begin(module, event):
    module.begin(event)
    assert all(t in ["Total", "AbsoluteStat"] for t in module.jes_sources)
    assert all(t in ["Total", "AbsoluteStat"] for t in event.JetSources)

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "jpt":    [[], [16.], [60., 70.]],
            "jeta":   [[], [-3.1], [0.5, 3.1]],
            "jphi":   [[], [0.1], [0.3, 0.5]],
            "gjpt":   [[], [17.], [65., 75.]],
            "gjeta":  [[], [-3.11], [1.0, 3.09]],
            "gjphi":  [[], [0.1], [0.3, 0.5]],
            "rho":    [0., 20., 50.],
            "met":    [200., 220., 240.],
            "mephi":  [0.9, 0.1, -0.7],
        }, {
            "jpt":         [[], [15.672], [76.2953858499882, 68.36000000000003]],
            "jptres":      [[], [0.456227396877664], [0.131449148093119, 0.179063214239771]],
            "jjersf":      [[], [0.9795], [1.271589764166470, 0.976571428571429]],
            "jjersfdown":  [[], [0.00797600816743227], [-0.061708613189990, 0.009142773551785]],
            "jjersfup":    [[], [-0.00797600816743238], [0.050076915810078, -0.009142773551785]],
            "jjestotdown": [[], [-0.07725], [-0.0144, -0.05163]],
            "jjestotup":   [[], [0.07725], [0.0144, 0.05163]],
            "met":         [200., 220.328, 232.10981279053],
            "mephi":       [0.9, 0.1, -0.75251459154],
        }],
    )
)
def test_jec_variations_event(module, event, inputs, outputs):
    def norm(*args, **kwargs):
        return 0.5
    np.random.normal = mock.Mock(side_effect=norm)
    event.config.dataset.idx = 2

    jet_pt = awk.JaggedArray.fromiter(inputs["jpt"]).astype(np.float32)
    jet_eta = awk.JaggedArray.fromiter(inputs["jeta"]).astype(np.float32)
    jet_phi = awk.JaggedArray.fromiter(inputs["jphi"]).astype(np.float32)

    genjet_pt = awk.JaggedArray.fromiter(inputs["gjpt"]).astype(np.float32)
    genjet_eta = awk.JaggedArray.fromiter(inputs["gjeta"]).astype(np.float32)
    genjet_phi = awk.JaggedArray.fromiter(inputs["gjphi"]).astype(np.float32)

    rho = np.array(inputs["rho"], dtype=np.float32)
    met_pt = np.array(inputs["met"], dtype=np.float32)
    met_phi = np.array(inputs["mephi"], dtype=np.float32)

    event.Jet_pt = jet_pt
    event.Jet.pt = jet_pt
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
    event.MET_phi = met_phi
    event.MET.phi = met_phi

    module.begin(event)
    module.event(event)

    assert np.allclose(
        event.Jet_ptResolution.content,
        awk.JaggedArray.fromiter(outputs["jptres"]).astype(np.float32).content,
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_JECjerSF.content,
        awk.JaggedArray.fromiter(outputs["jjersf"]).astype(np.float32).content,
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_JECjerSFDown.content,
        awk.JaggedArray.fromiter(outputs["jjersfdown"]).astype(np.float32).content,
        rtol=1e-5, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_JECjerSFUp.content,
        awk.JaggedArray.fromiter(outputs["jjersfup"]).astype(np.float32).content,
        rtol=1e-5, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_pt.content,
        awk.JaggedArray.fromiter(outputs["jpt"]).astype(np.float32).content,
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.MET_pt,
        np.array(outputs["met"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.MET_phi,
        np.array(outputs["mephi"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_JECjesTotalUp.content,
        awk.JaggedArray.fromiter(outputs["jjestotup"]).astype(np.float32).content,
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        event.Jet_JECjesTotalDown.content,
        awk.JaggedArray.fromiter(outputs["jjestotdown"]).astype(np.float32).content,
        rtol=1e-6, equal_nan=True,
    )

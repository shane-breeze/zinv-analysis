import pytest
import mock
import os
import numpy as np
import awkward as awk

from sequence.Readers import JecVariations

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.nsig = 0
        self.source = ''
        self.cache = {}
        self.variation_sources = ["jesTotal", "jerSF", "unclust", "jesAbsoluteStat"]

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
    )

def test_jec_variations_begin(module, event):
    module.begin(event)
    assert all(t in ["Total", "AbsoluteStat"] for t in module.jes_sources)
    assert all(t in ["Total", "AbsoluteStat"] for t in event.JetSources)

def test_jec_variations_event_empty(module, event):
    jet_pt = awk.JaggedArray(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float32),
    )
    jet_eta = awk.JaggedArray(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float32),
    )
    rho = np.array([], dtype=np.float32)

    jet_genjetidx = awk.JaggedArray(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
    )
    genjet_pt = awk.JaggedArray(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float32),
    )
    met_pt = np.array([], dtype=np.float32)
    met_phi = np.array([], dtype=np.float32)
    def jet_ptshift(self):
        return awk.JaggedArray(
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
        )
    jet_phi = awk.JaggedArray(
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float32),
    )

    event.Jet_pt = jet_pt
    event.Jet_eta = jet_eta
    event.fixedGridRhoFastjetAll = rho

    event.Jet_genJetIdx = jet_genjetidx
    event.GenJet_pt = genjet_pt
    event.MET_pt = met_pt
    event.MET_phi = met_phi
    event.Jet_ptShift = mock.Mock(side_effect=jet_ptshift)
    event.Jet_phi = jet_phi

    module.begin(event)
    module.event(event)

    assert np.array_equal(event.Jet_ptResolution, awk.JaggedArray([], [], []))

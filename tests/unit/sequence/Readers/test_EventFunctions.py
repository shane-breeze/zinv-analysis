import pytest
import mock
import numpy as np
import awkward as awk

from sequence.Readers import EventFunctions

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.nsig = 0
        self.source = ''
        self.cache = {}
        self.attribute_variation_sources = []

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return EventFunctions()

def test_metnox(event, module):
    module.begin(event)

    def met_ptshift(self):
        return np.array([95., 203.], dtype=np.float32)
    event.MET_ptShift = mock.Mock(side_effect=met_ptshift)

    def met_phishift(self):
        return np.array([0.3, 0.5], dtype=np.float32)
    event.MET_phiShift = mock.Mock(side_effect=met_phishift)

    def muon_selection(self, attr):
        if attr == 'phi':
            contents = np.array([-0.2, 0.9], dtype=np.float32)
        else:
            contents = np.array([40., 201.], dtype=np.float32)
        return awk.JaggedArray(
            np.array([0, 1], dtype=np.int32),
            np.array([1, 2], dtype=np.int32),
            contents,
        )
    event.MuonSelection = mock.Mock(side_effect=muon_selection)

    def ele_selection(self, attr):
        if attr == 'phi':
            contents = np.array([-1.5, -1.3], dtype=np.float32)
        else:
            contents = np.array([26., 91.], dtype=np.float32)
        return awk.JaggedArray(
            np.array([0, 0], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
            contents,
        )
    event.ElectronSelection = mock.Mock(side_effect=ele_selection)

    metnox_pt = event.METnoX_pt(event)
    metnox_phi = event.METnoX_phi(event)

    # Check arrays match with 0.0001% (1e-6)
    assert np.allclose(
        metnox_pt,
        np.array([131.5090395, 358.2540117], dtype=np.float32),
        rtol = 1e-6,
    )
    assert np.allclose(
        metnox_phi,
        np.array([0.1536553484, 0.4049836823], dtype=np.float32),
        rtol = 1e-6,
    )

def test_mindphi(event, module):
    module.begin(event)

    def metnox_phi(self):
        return np.array([0.0, 0.5, 0.9, 1.3, 1.7, 2.1], dtype=np.float32)
    event.METnoX_phi = mock.Mock(side_effect=metnox_phi)

    def jet_selection(self, attr):
        assert attr == 'phi'
        return awk.JaggedArray(
            np.array([0, 0, 1, 3, 6, 10], dtype=np.int32),
            np.array([0, 1, 3, 6, 10, 15], dtype=np.int32),
            np.array([
                0.0,
                0.0, 0.5,
                0.0, 0.5, 1.0,
                0.0, 0.5, 1.0, 1.5,
                0.5, 1.0, 1.5, 2.0, 2.5,
                0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
            ], dtype=np.float32),
        )
    event.JetSelection = mock.Mock(side_effect=jet_selection)

    assert np.allclose(
        event.MinDPhiJ1234METnoX(event),
        np.array([np.nan, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        rtol = 1e-6, equal_nan=True,
    )

def test_met_dcalo(event, module):
    module.begin(event)
    event.CaloMET_pt = np.array([50., 75., 150., 150.], dtype=np.float32)

    def met_ptshift(self):
        return np.array([100., 200., 150., 150.], dtype=np.float32)
    event.MET_ptShift = mock.Mock(side_effect=met_ptshift)

    def metnox_pt(self):
        return np.array([250., 0., 200., 0.], dtype=np.float32)
    event.METnoX_pt = mock.Mock(side_effect=metnox_pt)

    assert np.allclose(
        event.MET_dCaloMET(event),
        np.array([0.2, np.inf, 0., np.nan], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

def test_mtw(event, module):
    module.begin(event)

    def met_ptshift(self):
        return np.array([100., 200., 300., 400., 500.], dtype=np.float32)
    def met_phishift(self):
        return np.array([-1., 0.5, 0., 0.5, 1.], dtype=np.float32)
    def muon_selection(self, attr):
        if attr == "ptShift":
            content = np.array([50., 60., 70., 80.], dtype=np.float32)
        elif attr == "phi":
            content = np.array([0.1, -0.1, 0.3, -0.5], dtype=np.float32)
        else:
            assert False

        return awk.JaggedArray(
            np.array([0, 0, 1, 3, 3], dtype=np.int32),
            np.array([0, 1, 3, 3, 4], dtype=np.int32),
            content,
        )
    def ele_selection(self, attr):
        if attr == "ptShift":
            content = np.array([40., 45.], dtype=np.float32)
        elif attr == "phi":
            content = np.array([2.1, -2.1], dtype=np.float32)
        else:
            assert False

        return awk.JaggedArray(
            np.array([0, 0, 0, 0, 1], dtype=np.int32),
            np.array([0, 0, 0, 1, 2], dtype=np.int32),
            content,
        )

    event.MET_ptShift = mock.Mock(side_effect=met_ptshift)
    event.MET_phiShift = mock.Mock(side_effect=met_phishift)
    event.MuonSelection = mock.Mock(side_effect=muon_selection)
    event.ElectronSelection = mock.Mock(side_effect=ele_selection)

    assert np.allclose(
        event.MTW(event),
        np.array([np.nan, 39.73386616, np.nan, 181.4783313, np.nan], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

def test_mll(event, module):
    module.begin(event)
    event.size = 4

    def muon_selection(self, attr):
        if attr == "ptShift":
            content = np.array([50., 60., 70., 80.], dtype=np.float32)
        elif attr == "eta":
            content = np.array([-2.1, 1.4, -0.5, 0.6], dtype=np.float32)
        elif attr == "phi":
            content = np.array([0.1, -0.1, 0.3, -0.5], dtype=np.float32)
        elif attr == "mass":
            content = np.array([0., 0., 0., 0.], dtype=np.float32)
        else:
            assert False

        return awk.JaggedArray(
            np.array([0, 0, 2, 2], dtype=np.int32),
            np.array([0, 2, 2, 4], dtype=np.int32),
            content,
        )
    def ele_selection(self, attr):
        if attr == "ptShift":
            content = np.array([40., 45., 50., 55.], dtype=np.float32)
        elif attr == "eta":
            content = np.array([0.4, 0.8, 1.2, 1.6], dtype=np.float32)
        elif attr == "phi":
            content = np.array([2.1, -2.1, 0.2, -0.2], dtype=np.float32)
        elif attr == "mass":
            content = np.array([0., 0., 0., 0.], dtype=np.float32)
        else:
            assert False

        return awk.JaggedArray(
            np.array([0, 0, 0, 2], dtype=np.int32),
            np.array([0, 0, 2, 4], dtype=np.int32),
            content,
        )

    event.MuonSelection = mock.Mock(side_effect=muon_selection)
    event.ElectronSelection = mock.Mock(side_effect=ele_selection)

    assert np.allclose(
        event.MLL(event),
        np.array([np.nan, 305.8701498, 75.21169786, np.nan], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

def test_lepton_charge(event, module):
    module.begin(event)
    event.size = 4

    def muon_selection(self, attr):
        assert attr == 'charge'
        return awk.JaggedArray(
            np.array([0, 0, 1, 1], dtype=np.int32),
            np.array([0, 1, 1, 2], dtype=np.int32),
            np.array([1, -1], dtype=np.int32),
        )
    def ele_selection(self, attr):
        assert attr == 'charge'
        return awk.JaggedArray(
            np.array([0, 0, 0, 1], dtype=np.int32),
            np.array([0, 0, 1, 2], dtype=np.int32),
            np.array([-1, 1], dtype=np.int32),
        )

    event.MuonSelection = mock.Mock(side_effect=muon_selection)
    event.ElectronSelection = mock.Mock(side_effect=ele_selection)

    assert np.allclose(
        event.LeptonCharge(event),
        np.array([0, 1, -1, 0], dtype=np.int32),
        rtol=1e-6, equal_nan=True,
    )

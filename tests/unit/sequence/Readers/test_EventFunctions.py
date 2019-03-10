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

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return EventFunctions()

@pytest.fixture()
def module_begin(event, module):
    module.begin(event)
    return module

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

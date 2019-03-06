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

    event.MET_ptShift = mock.Mock(side_effect=np.array([95., 203.], dtype=float))
    event.MET_phiShift = mock.Mock(side_effect=np.array([0.3, 0.5], dtype=float))

    def muon_selection(self, attr):
        if attr == 'phi':
            contents = np.array([-0.2, 0.9], dtype=float)
        else:
            contents = np.array([40., 201.], dtype=float)
        return awk.JaggedArray(
            np.array([0, 1], dtype=int),
            np.array([1, 2], dtype=int),
            contents,
        )
    event.MuonSelection = mock.Mock(side_effect=muon_selection)

    def ele_selection(self, attr):
        if attr == 'phi':
            contents = np.array([-1.5, -1.3], dtype=float)
        else:
            contents = np.array([26., 91.], dtype=float)
        return awk.JaggedArray(
            np.array([0, 0], dtype=int),
            np.array([0, 2], dtype=int),
            contents,
        )
    event.ElectronSelection = mock.Mock(side_effect=ele_selection)

    print(event.METnoX_pt(event))
    assert False

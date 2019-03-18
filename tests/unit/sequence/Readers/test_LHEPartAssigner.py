import pytest
import mock
import numpy as np
import awkward as awk

from sequence.Readers import LHEPartAssigner, GenPartAssigner

class DummyEvent(object):
    def __init__(self):
        self.config = mock.MagicMock()
        self.LHEPart = mock.MagicMock()
        self.GenPart = mock.MagicMock()
        self.delete_branches = mock.MagicMock()

@pytest.fixture()
def event():
    ev = DummyEvent()
    ev.size = 1
    ev.config.dataset.parent = "DummyParent"
    one_array = awk.JaggedArray(
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
    )
    ev.LHEPart.pdgId = one_array[:,:]
    ev.GenPart.statusFlags = one_array[:,:]
    ev.GenPart.pdgId = one_array[:,:]
    return ev

@pytest.fixture()
def lhe_module():
    return LHEPartAssigner(old_parents=["DummyParent"])

@pytest.fixture()
def gen_module():
    return GenPartAssigner(old_parents=["DummyParent"])

@pytest.mark.parametrize("parent,result", (
    ["DummyParent", None],
    ["NotDummyParent", True],
))
def test_lhe_module_parent(lhe_module, event, parent, result):
    event.config.dataset.parent = parent
    assert lhe_module.event(event) == result

@pytest.mark.parametrize("parent,result", (
    ["DummyParent", None],
    ["NotDummyParent", True],
))
def test_gen_module_parent(gen_module, event, parent, result):
    event.config.dataset.parent = parent
    assert gen_module.event(event) == result

params1 = [{
    "starts": [0, 1, 4, 5, 6, 7, 8],
    "stops":  [1, 4, 5, 6, 7, 8, 9],
    "pdgs":   [10, 10, 11, 13, -11, 13, -13, 15, -15],
    "isele":  [0, 1, 1, 0, 0, 0, 0],
    "ismu":   [0, 0, 0, 1, 1, 0, 0],
    "istau":  [0, 0, 0, 0, 0, 1, 1],
}]
@pytest.mark.parametrize(
    ",".join(params1[0].keys()), (
        [ps[k] for k in params1[0].keys()]
        for ps in params1
    )
)
def test_lhe_module(lhe_module, event, starts, stops, pdgs, isele, ismu, istau):
    event.size = len(starts)
    event.LHEPart.pdgId = awk.JaggedArray(
        np.array(starts, dtype=np.int32),
        np.array(stops, dtype=np.int32),
        np.array(pdgs, dtype=np.int32),
    )
    lhe_module.event(event)
    assert np.array_equal(event.LeptonIsElectron, np.array(isele, dtype=np.bool8))
    assert np.array_equal(event.LeptonIsMuon, np.array(ismu, dtype=np.bool8))
    assert np.array_equal(event.LeptonIsTau, np.array(istau, dtype=np.bool8))

params2 = [{
    "starts": [0, 1, 4],
    "stops":  [1, 4, 7],
    "flags":  [0, 1, 2, 1<<10, 1<<10, 1<<10, 0],
    "pdgs":   [11, 13, -11, -13, -11, 0, 11],
    "ngtaul": [0, 1, 1],
}]
@pytest.mark.parametrize(
    ",".join(params2[0].keys()), (
        [ps[k] for k in params2[0].keys()]
        for ps in params2
    )
)
def test_gen_module(gen_module, event, starts, stops, flags, pdgs, ngtaul):
    event.size = len(starts)
    event.GenPart.statusFlags = awk.JaggedArray(
        np.array(starts, dtype=np.int32),
        np.array(stops, dtype=np.int32),
        np.array(flags, dtype=np.int32),
    )
    event.GenPart.pdgId = awk.JaggedArray(
        np.array(starts, dtype=np.int32),
        np.array(stops, dtype=np.int32),
        np.array(pdgs, dtype=np.int32),
    )
    gen_module.event(event)
    assert np.array_equal(event.nGenTauL, np.array(ngtaul, dtype=np.int32))

def test_lhe_module_delete(lhe_module, event):
    lhe_module.event(event)
    assert event.delete_branches.assert_called

def test_gen_module_delete(gen_module, event):
    gen_module.event(event)
    assert event.delete_branches.assert_called

import pytest
import mock
import numpy as np
import awkward as awk

from sequence.Readers import ObjectFunctions

class DummyColl(object):
    def __init__(self):
        pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.nsig = 0
        self.source = ''
        self.cache = {}

        self.Jet = DummyColl()
        self.Muon = DummyColl()

@pytest.fixture()
def event():
    return DummyEvent()

selections = [
    ("Base1", "Derivative1", True),
    ("Base1", "Derivative2", True),
    ("Base2", "Derivative3", False),
]

@pytest.fixture()
def module():
    return ObjectFunctions(selections=selections)

def test_objfunc_init(module):
    assert module.selections == selections

def test_objfunc_begin(module, event):
    assert module.begin(event) is None
    assert hasattr(event, "Jet_ptShift")
    assert hasattr(event, "Muon_ptShift")
    assert hasattr(event, "Electron_ptShift")
    assert hasattr(event, "Photon_ptShift")
    assert hasattr(event, "Tau_ptShift")
    assert hasattr(event, "MET_ptShift")
    assert hasattr(event, "MET_phiShift")

    for objname, selection, xclean in selections:
        assert hasattr(event, selection)
        if xclean:
            assert hasattr(event, selection+"NoXClean")

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "nsig":   0,
            "source": '',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "jpt":    [20., 40., 60., 80.],
            "evvars": {},
        }, {
            "jptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   1,
            "source": 'Var1',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "jpt":    [20., 40., 60., 80.],
            "evvars": {
                "JECVar1Up": [0.1, 0.2, 0.3, 0.4],
                "JECVar1Down": [-0.05, 0.1, -0.6, -0.9],
            },
        }, {
            "jptshift": [20.*1.1, 40.*1.2, 60.*1.3, 80.*1.4],
        }], [{
            "nsig":   -1,
            "source": 'Var1',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "jpt":    [20., 40., 60., 80.],
            "evvars": {
                "JECVar1Up": [0.1, 0.2, 0.3, 0.4],
                "JECVar1Down": [-0.05, 0.1, -0.6, -0.9],
            },
        }, {
            "jptshift": [20.*0.95, 40.*1.1, 60.*0.4, 80.*0.1],
        }], [{
            "nsig":   1,
            "source": 'Var1',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "jpt":    [20., 40., 60., 80.],
            "evvars": {
                "JECVar1Up": [0.1, 0.2, 0.3, 0.4],
            },
        }, {
            "jptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   1,
            "source": 'Var1',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "jpt":    [20., 40., 60., 80.],
            "evvars": {
                "JECVar2Up": [0.1, 0.2, 0.3, 0.4],
                "JECVar2Down": [-0.05, 0.1, -0.6, -0.9],
            },
        }, {
            "jptshift": [20., 40., 60., 80.],
        }],
    ),
)
def test_objfunc_jptshift(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    event.Jet.pt = awk.JaggedArray(
        inputs["starts"], inputs["stops"], inputs["jpt"],
    )

    for key, val in inputs["evvars"].items():
        setattr(event.Jet, key, awk.JaggedArray(
            inputs["starts"], inputs["stops"], val,
        ))

    module.begin(event)
    jptshift = event.Jet_ptShift(event)

    assert np.array_equal(jptshift.starts, np.array(inputs["starts"]))
    assert np.array_equal(jptshift.stops, np.array(inputs["stops"]))
    assert np.allclose(jptshift.content, np.array(outputs["jptshift"]))

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "nsig":   0,
            "source": '',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "mpt":    [20., 40., 60., 80.],
            "evvars": {
                "ptErr": [1., 2., 4., 8.],
            },
        }, {
            "mptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   1,
            "source": 'muonPtScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "mpt":    [20., 40., 60., 80.],
            "evvars": {
                "ptErr": [1., 2., 4., 8.],
            },
        }, {
            "mptshift": [21., 42., 64., 88.],
        }], [{
            "nsig":   -1,
            "source": 'muonPtScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "mpt":    [20., 40., 60., 80.],
            "evvars": {
                "ptErr": [1., 2., 4., 8.],
            },
        }, {
            "mptshift": [19., 38., 56., 72.],
        }], [{
            "nsig":   -1,
            "source": 'someRandomThing',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "mpt":    [20., 40., 60., 80.],
            "evvars": {
                "ptErr": [1., 2., 4., 8.],
            },
        }, {
            "mptshift": [20., 40., 60., 80.],
        }],
    ),
)
def test_objfunc_muptshift(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    event.Muon.pt = awk.JaggedArray(
        inputs["starts"], inputs["stops"], inputs["mpt"],
    )

    for key, val in inputs["evvars"].items():
        jagarr = awk.JaggedArray(inputs["starts"], inputs["stops"], val)
        setattr(event.Muon, key, jagarr)
        setattr(event, "Muon_{}".format(key), jagarr)

    module.begin(event)
    mptshift = event.Muon_ptShift(event)

    assert np.array_equal(mptshift.starts, np.array(inputs["starts"]))
    assert np.array_equal(mptshift.stops, np.array(inputs["stops"]))
    assert np.allclose(mptshift.content, np.array(outputs["mptshift"]))

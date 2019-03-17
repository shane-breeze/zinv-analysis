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
        self.attribute_variation_sources = [
            "Var1", "muonPtScale", "eleEnergyScale", "photonEnergyScale",
        ]

        self.Jet = DummyColl()
        self.Muon = DummyColl()
        self.Electron = DummyColl()
        self.Photon = DummyColl()
        self.Tau = DummyColl()
        self.MET = DummyColl()

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
    return ObjectFunctions(selections=selections, unclust_threshold=15.)

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
        inputs["starts"], inputs["stops"], np.array(inputs["jpt"], dtype=np.float32),
    )

    for key, val in inputs["evvars"].items():
        setattr(event.Jet, key, awk.JaggedArray(
            inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32),
        ))

    module.begin(event)
    jptshift = event.Jet_ptShift(event)

    assert np.array_equal(jptshift.starts, np.array(inputs["starts"]))
    assert np.array_equal(jptshift.stops, np.array(inputs["stops"]))
    assert np.allclose(
        jptshift.content,
        np.array(outputs["jptshift"]),
        rtol=1e-6, equal_nan=True,
    )

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
        inputs["starts"], inputs["stops"], np.array(inputs["mpt"], dtype=np.float32),
    )

    for key, val in inputs["evvars"].items():
        jagarr = awk.JaggedArray(inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32))
        setattr(event.Muon, key, jagarr)
        setattr(event, "Muon_{}".format(key), jagarr)

    module.begin(event)
    mptshift = event.Muon_ptShift(event)

    assert np.array_equal(mptshift.starts, np.array(inputs["starts"]))
    assert np.array_equal(mptshift.stops, np.array(inputs["stops"]))
    assert np.allclose(
        mptshift.content,
        np.array(outputs["mptshift"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "nsig":   0,
            "source": '',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ept":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "eptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   1,
            "source": 'eleEnergyScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ept":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "eptshift": [21., 42., 64., 88.],
        }], [{
            "nsig":   -1,
            "source": 'eleEnergyScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ept":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "eptshift": [19., 38., 56., 72.],
        }], [{
            "nsig":   -1,
            "source": 'someRandomThing',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ept":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "eptshift": [20., 40., 60., 80.],
        }],
    ),
)
def test_objfunc_eptshift(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    event.Electron.pt = awk.JaggedArray(
        inputs["starts"], inputs["stops"], np.array(inputs["ept"], dtype=np.float32),
    )

    for key, val in inputs["evvars"].items():
        jagarr = awk.JaggedArray(inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32))
        setattr(event.Electron, key, jagarr)
        setattr(event, "Electron_{}".format(key), jagarr)

    module.begin(event)
    eptshift = event.Electron_ptShift(event)

    assert np.array_equal(eptshift.starts, np.array(inputs["starts"]))
    assert np.array_equal(eptshift.stops, np.array(inputs["stops"]))
    assert np.allclose(
        eptshift.content,
        np.array(outputs["eptshift"]),
        rtol=1e-6, equal_nan=True,
    )

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "nsig":   0,
            "source": '',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ypt":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "yptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   1,
            "source": 'photonEnergyScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ypt":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "yptshift": [21., 42., 64., 88.],
        }], [{
            "nsig":   -1,
            "source": 'photonEnergyScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ypt":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "yptshift": [19., 38., 56., 72.],
        }], [{
            "nsig":   -1,
            "source": 'someRandomThing',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ypt":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "yptshift": [20., 40., 60., 80.],
        }],
    ),
)
def test_objfunc_yptshift(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    event.Photon.pt = awk.JaggedArray(
        inputs["starts"], inputs["stops"], np.array(inputs["ypt"], dtype=np.float32),
    )

    for key, val in inputs["evvars"].items():
        jagarr = awk.JaggedArray(inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32))
        setattr(event.Photon, key, jagarr)
        setattr(event, "Photon_{}".format(key), jagarr)

    module.begin(event)
    yptshift = event.Photon_ptShift(event)

    assert np.array_equal(yptshift.starts, np.array(inputs["starts"]))
    assert np.array_equal(yptshift.stops, np.array(inputs["stops"]))
    assert np.allclose(
        yptshift.content,
        np.array(outputs["yptshift"]),
        rtol=1e-6, equal_nan=True,
    )

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "nsig":   0,
            "source": '',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "tpt":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "tptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   1,
            "source": 'tauEnergyScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "tpt":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "tptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   -1,
            "source": 'tauEnergyScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "tpt":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "tptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   -1,
            "source": 'someRandomThing',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "tpt":    [20., 40., 60., 80.],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "tptshift": [20., 40., 60., 80.],
        }],
    ),
)
def test_objfunc_tptshift(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    tpt = awk.JaggedArray(inputs["starts"], inputs["stops"], np.array(inputs["tpt"], dtype=np.float32))
    event.Tau_pt = tpt
    event.Tau.pt = tpt

    for key, val in inputs["evvars"].items():
        jagarr = awk.JaggedArray(inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32))
        setattr(event.Tau, key, jagarr)
        setattr(event, "Tau_{}".format(key), jagarr)

    module.begin(event)
    tptshift = event.Tau_ptShift(event)

    assert np.array_equal(tptshift.starts, np.array(inputs["starts"]))
    assert np.array_equal(tptshift.stops, np.array(inputs["stops"]))
    assert np.allclose(
        tptshift.content,
        np.array(outputs["tptshift"]),
        rtol=1e-6, equal_nan=True,
    )

#@pytest.mark.parametrize(
#    "inputs,outputs", (
#        [{
#        #    "nsig":     0,
#        #    "source":   '',
#        #    "met":      [200., 300., 400.],
#        #    "mephi":    [0.3, 0.6, 0.9],
#        #    "jstarts":  [0, 1, 2],
#        #    "jstops":   [1, 2, 4],
#        #    "jpt":      [20., 40., 60., 80.],
#        #    "jptshift": [20., 40., 60., 80.],
#        #    "jphi":     [-0.3, 0.8, -1.1, 2.5],
#        #    "evvars": {
#        #        "MetUnclustEnUpDeltaX": [10., 15., 25.],
#        #        "MetUnclustEnUpDeltaY": [5., 7., 13.],
#        #    },
#        #}, {
#        #    "metshift": [200., 300., 400.],
#        #}], [{
#            "nsig":     1,
#            "source":   'Var1',
#            "met":      [200., 300., 400.],
#            "mephi":    [0.3, 0.6, 0.9],
#            "jstarts":  [0, 1, 2],
#            "jstops":   [1, 2, 4],
#            "jpt":      [16., 40., 60., 80.],
#            "jptshift": [14., 80., 45., 121.],
#            "jphi":     [-0.3, 0.8, -1.1, 2.5],
#            "evvars": {
#                "MetUnclustEnUpDeltaX": [10., 15., 25.],
#                "MetUnclustEnUpDeltaY": [5., 7., 13.],
#            },
#        }, {
#            "metshift": [200., 300., 400.],
#        }],
#    )
#)
#def test_objfunc_metshift(module, event, inputs, outputs):
#    event.nsig = inputs["nsig"]
#    event.source = inputs["source"]
#
#    met = np.array(inputs["met"], dtype=np.float32)
#    mephi = np.array(inputs["mephi"], dtype=np.float32)
#    event.MET_pt = met
#    event.MET.pt = met
#    event.MET_phi = mephi
#    event.MET.phi = mephi
#
#    starts, stops = inputs["jstarts"], inputs["jstops"]
#    jpt = awk.JaggedArray(starts, stops, inputs["jpt"])
#    jptshift = awk.JaggedArray(starts, stops, inputs["jptshift"])
#    jphi = awk.JaggedArray(starts, stops, inputs["jphi"])
#    event.Jet.pt = jpt
#    event.Jet_pt = jpt
#    event.Jet.phi = jphi
#    event.Jet_phi = jphi
#
#    for key, val in inputs["evvars"].items():
#        setattr(event.MET, key, np.array(val, dtype=np.float32))
#        setattr(event, "MET_"+key, np.array(val, dtype=np.float32))
#
#    module.begin(event)
#    metshift = event.MET_ptShift(event)
#
#    assert np.allclose(
#        metshift,
#        np.array(outputs["metshift"]),
#        rtol=1e-6, equal_nan=True,
#    )
#    assert False

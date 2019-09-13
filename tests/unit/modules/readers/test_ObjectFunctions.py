import pytest
import mock
import numpy as np
import awkward as awk
import collections

from zinv.modules.readers import ObjectFunctions

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
            "jesTotal", "jerSF", "unclust",
        ]

        self.Jet = DummyColl()
        self.Muon = DummyColl()
        self.Electron = DummyColl()
        self.Photon = DummyColl()
        self.Tau = DummyColl()
        self.MET = DummyColl()

    def register_function(self, event, name, function):
        self.__dict__[name] = function

    def hasbranch(self, branch):
        return hasattr(self, branch)

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
            "source": 'jesVar1',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "jpt":    [20., 40., 60., 80.],
            "evvars": {
                "jesVar1Up": [0.1, 0.2, 0.3, 0.4],
                "jesVar1Down": [-0.05, 0.1, -0.6, -0.9],
            },
        }, {
            "jptshift": [20.*1.1, 40.*1.2, 60.*1.3, 80.*1.4],
        }], [{
            "nsig":   -1,
            "source": 'jesVar1',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "jpt":    [20., 40., 60., 80.],
            "evvars": {
                "jesVar1Up": [0.1, 0.2, 0.3, 0.4],
                "jesVar1Down": [-0.05, 0.1, -0.6, -0.9],
            },
        }, {
            "jptshift": [20.*0.95, 40.*1.1, 60.*0.4, 80.*0.1],
        }], [{
            "nsig":   1,
            "source": 'jesVar1',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "jpt":    [20., 40., 60., 80.],
            "evvars": {
                "jesVar2Up": [0.1, 0.2, 0.3, 0.4],
                "jesVar2Down": [-0.05, 0.1, -0.6, -0.9],
            },
        }, {
            "jptshift": [20., 40., 60., 80.],
        }],
    ),
)
def test_objfunc_jptshift(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    jet_pt = awk.JaggedArray(
        inputs["starts"], inputs["stops"], np.array(inputs["jpt"], dtype=np.float32),
    )
    event.Jet_pt = jet_pt
    event.Jet.pt = jet_pt

    curr_source = ""
    arr_up = awk.JaggedArray(
        inputs["starts"], inputs["stops"], np.zeros_like(inputs["jpt"], dtype=np.float32),
    )
    arr_down = awk.JaggedArray(
        inputs["starts"], inputs["stops"], np.zeros_like(inputs["jpt"], dtype=np.float32),
    )
    for key, val in inputs["evvars"].items():
        curr_source = key.replace("Up", "").replace("Down", "")
        if key.endswith("Up"):
            arr_up = awk.JaggedArray(
                inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32),
            )
        elif key.endswith("Down"):
            arr_down = awk.JaggedArray(
                inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32),
            )

    jer_sf = lambda ev, source, nsig: (
        (arr_up if nsig>=0. else arr_down) if source==curr_source else
        awk.JaggedArray(inputs["starts"], inputs["stops"], np.zeros_like(inputs["jpt"], dtype=np.float32))
    )
    event.Jet_jesSF = jer_sf
    event.Jet.jesSF = jer_sf

    module.begin(event)
    jptshift = event.Jet_ptShift(event, event.source, event.nsig)

    print(jptshift.content)
    print(np.array(outputs["jptshift"]))

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
            "jphi": [[0.4, 0.5], [], [1., 1.2]],
            "mephi": [0.3, 0.6, 0.9],
            "source": "",
            "nsig": 0.,
        }, {
            "jdphimet": [[0.1, 0.2], [], [0.1, 0.3]],
        }], [{
            "jphi": [[0.4, 0.5], [], [1., 1.2]],
            "mephi": [0.3, 0.6, 0.9],
            "source": "jesTotal",
            "nsig": 1.,
        }, {
            "jdphimet": [[0.1, 0.2], [], [0.1, 0.3]],
        }],
    )
)
def test_jet_dphimet(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    module.begin(event)

    mephi = np.array(inputs["mephi"], dtype=np.float32)
    event.MET.phiShift = mock.Mock(side_effect=lambda ev, source, nsig: mephi)
    event.MET_phiShift = mock.Mock(side_effect=lambda ev, source, nsig: mephi)

    jphi = awk.JaggedArray.fromiter(inputs["jphi"]).astype(np.float32)
    event.Jet.phi = jphi
    event.Jet_phi = jphi

    jdphimet = event.Jet_dphiMET(event, event.source, event.nsig)
    ojdphimet = awk.JaggedArray.fromiter(outputs["jdphimet"]).astype(np.float32)
    assert np.array_equal(jdphimet.starts, ojdphimet.starts)
    assert np.array_equal(jdphimet.stops, ojdphimet.stops)
    assert np.allclose(
        jdphimet.content,
        ojdphimet.content,
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
            "meta":   [-2.3, -2.0, 0., 2.5],
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
            "meta":   [-2.3, -2.0, 0., 2.5],
            "evvars": {
                "ptErr": [1., 2., 4., 8.],
            },
        }, {
            "mptshift": [20.54, 40.36, 60.24, 81.36],
        }], [{
            "nsig":   -1,
            "source": 'muonPtScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "mpt":    [20., 40., 60., 80.],
            "meta":   [-2.3, -2.0, 0., 2.5],
            "evvars": {
                "ptErr": [1., 2., 4., 8.],
            },
        }, {
            "mptshift": [19.46, 39.64, 59.76, 78.64],
        }], [{
            "nsig":   -1,
            "source": 'someRandomThing',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "mpt":    [20., 40., 60., 80.],
            "meta":   [-2.3, -2.0, 0., 2.5],
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
    event.Muon.eta = awk.JaggedArray(
        inputs["starts"], inputs["stops"], np.array(inputs["meta"], dtype=np.float32),
    )

    for key, val in inputs["evvars"].items():
        jagarr = awk.JaggedArray(inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32))
        setattr(event.Muon, key, jagarr)
        setattr(event, "Muon_{}".format(key), jagarr)

    module.begin(event)
    mptshift = event.Muon_ptShift(event, event.source, event.nsig)

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
            "eeta":   [0., 1., 2., 3.],
            "evvars": {
                "energyErr": [1., 3.08616127, 15.04878276, 80.54129597],
            },
        }, {
            "eptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   1,
            "source": 'eleEnergyScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ept":    [20., 40., 60., 80.],
            "eeta":   [0., 1., 2., 3.],
            "evvars": {
                "energyErr": [1., 3.08616127, 15.04878276, 80.54129597],
            },
        }, {
            "eptshift": [21., 42., 64., 88.],
        }], [{
            "nsig":   -1,
            "source": 'eleEnergyScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ept":    [20., 40., 60., 80.],
            "eeta":   [0., 1., 2., 3.],
            "evvars": {
                "energyErr": [1., 3.08616127, 15.04878276, 80.54129597],
            },
        }, {
            "eptshift": [19., 38., 56., 72.],
        }], [{
            "nsig":   -1,
            "source": 'someRandomThing',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ept":    [20., 40., 60., 80.],
            "eeta":   [0., 1., 2., 3.],
            "evvars": {
                "energyErr": [1., 3.08616127, 15.04878276, 80.54129597],
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
    event.Electron.eta = awk.JaggedArray(
        inputs["starts"], inputs["stops"], np.array(inputs["eeta"], dtype=np.float32),
    )

    for key, val in inputs["evvars"].items():
        jagarr = awk.JaggedArray(inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32))
        setattr(event.Electron, key, jagarr)
        setattr(event, "Electron_{}".format(key), jagarr)

    module.begin(event)
    eptshift = event.Electron_ptShift(event, event.source, event.nsig)

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
            "yeta":   [0., 1., 2., 3.],
            "evvars": {
                "energyErr": [1., 3.08616127, 15.04878276, 80.54129597],
            },
        }, {
            "yptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   1,
            "source": 'photonEnergyScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ypt":    [20., 40., 60., 80.],
            "yeta":   [0., 1., 2., 3.],
            "evvars": {
                "energyErr": [1., 3.08616127, 15.04878276, 80.54129597],
            },
        }, {
            "yptshift": [21., 42., 64., 88.],
        }], [{
            "nsig":   -1,
            "source": 'photonEnergyScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ypt":    [20., 40., 60., 80.],
            "yeta":   [0., 1., 2., 3.],
            "evvars": {
                "energyErr": [1., 3.08616127, 15.04878276, 80.54129597],
            },
        }, {
            "yptshift": [19., 38., 56., 72.],
        }], [{
            "nsig":   -1,
            "source": 'someRandomThing',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "ypt":    [20., 40., 60., 80.],
            "yeta":   [0., 1., 2., 3.],
            "evvars": {
                "energyErr": [1., 3.08616127, 15.04878276, 80.54129597],
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
    event.Photon.eta = awk.JaggedArray(
        inputs["starts"], inputs["stops"], np.array(inputs["yeta"], dtype=np.float32),
    )


    for key, val in inputs["evvars"].items():
        jagarr = awk.JaggedArray(inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32))
        setattr(event.Photon, key, jagarr)
        setattr(event, "Photon_{}".format(key), jagarr)

    module.begin(event)
    yptshift = event.Photon_ptShift(event, event.source, event.nsig)

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
            "tdm":    [0, 1, 2, 3],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "tptshift": [20., 40., 60., 80.],
        }], [{
            "nsig":   1,
            "source": 'tauPtScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "tpt":    [20., 40., 60., 80.],
            "tdm":    [0, 1, 2, 3],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "tptshift": [20.24, 40.4, 60.6, 80.],
        }], [{
            "nsig":   -1,
            "source": 'tauPtScale',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "tpt":    [20., 40., 60., 80.],
            "tdm":    [0, 1, 2, 3],
            "evvars": {
                "energyErr": [1., 2., 4., 8.],
            },
        }, {
            "tptshift": [19.76, 39.6, 59.4, 80.],
        }], [{
            "nsig":   -1,
            "source": 'someRandomThing',
            "starts": [0, 1, 2],
            "stops":  [1, 2, 4],
            "tpt":    [20., 40., 60., 80.],
            "tdm":    [0, 1, 2, 3],
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
    tdm = awk.JaggedArray(inputs["starts"], inputs["stops"], np.array(inputs["tdm"], dtype=np.float32))
    event.Tau_decayMode = tdm
    event.Tau.decayMode = tdm

    for key, val in inputs["evvars"].items():
        jagarr = awk.JaggedArray(inputs["starts"], inputs["stops"], np.array(val, dtype=np.float32))
        setattr(event.Tau, key, jagarr)
        setattr(event, "Tau_{}".format(key), jagarr)

    module.begin(event)
    tptshift = event.Tau_ptShift(event, event.source, event.nsig)

    assert np.array_equal(tptshift.starts, np.array(inputs["starts"]))
    assert np.array_equal(tptshift.stops, np.array(inputs["stops"]))
    assert np.allclose(
        tptshift.content,
        np.array(outputs["tptshift"]),
        rtol=1e-6, equal_nan=True,
    )

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "nsig":     0,
            "source":   '',
            "met":      [200., 300., 400.],
            "mephi":    [0.3, 0.6, 0.9],
            "jpt":      [[20.], [40.], [60., 80.]],
            "jptshift": [[20.], [40.], [60., 80.]],
            "jphi":     [[-0.3], [0.8], [-1.1, 2.5]],
            "evvars": {
                "MetUnclustEnUpDeltaX": [10., 15., 25.],
                "MetUnclustEnUpDeltaY": [5., 7., 13.],
            },
        }, {
            "metshift": [200., 300., 400.],
            "mephishift": [0.3, 0.6, 0.9],
        }], [{
            "nsig":     1,
            "source":   'Var1',
            "met":      [200., 300., 400.],
            "mephi":    [0.3, 0.6, 0.9],
            "jpt":      [[16.], [40.], [60., 80.]],
            "jptshift": [[14.], [80.], [45., 121.]],
            "jphi":     [[-0.3], [0.8], [-1.1, 2.5]],
            "evvars": {
                "MetUnclustEnUpDeltaX": [10., 15., 25.],
                "MetUnclustEnUpDeltaY": [5., 7., 13.],
            },
        }, {
            "metshift": [201.653833318208, 260.918382127075, 398.714177256099],
            "mephishift": [0.29439985429023, 0.569538357995496, 0.762572497967355],
        }], [{
            "nsig":     1,
            "source":   'unclust',
            "met":      [200., 300., 400.],
            "mephi":    [0.3, 0.6, 0.9],
            "jpt":      [[16.], [40.], [60., 80.]],
            "jptshift": [[16.], [40.], [60., 80.]],
            "jphi":     [[-0.3], [0.8], [-1.1, 2.5]],
            "evvars": {
                "MetUnclustEnUpDeltaX": [10., 15., 25.],
                "MetUnclustEnUpDeltaY": [5., 7., 13.],
            },
        }, {
            "metshift": [211.038826687946, 316.343988282449, 425.878855104992],
            "mephishift": [0.30863112737787, 0.591489263592354, 0.872988464087041],
        }], [{
            "nsig":     -1,
            "source":   'unclust',
            "met":      [200., 300., 400.],
            "mephi":    [0.3, 0.6, 0.9],
            "jpt":      [[16.], [40.], [60., 80.]],
            "jptshift": [[16.], [40.], [60., 80.]],
            "jphi":     [[-0.3], [0.8], [-1.1, 2.5]],
            "evvars": {
                "MetUnclustEnUpDeltaX": [10., 15., 25.],
                "MetUnclustEnUpDeltaY": [5., 7., 13.],
            },
        }, {
            "metshift": [188.977812534104, 283.680244425927, 374.453202382435],
            "mephishift": [0.290361256916289, 0.609490714518387, 0.930722271015859],
        }], [{
            "nsig":     1,
            "source":   'unclust',
            "met":      [200., 300., 400.],
            "mephi":    [0.3, 0.6, 0.9],
            "jpt":      [[16.], [40.], [60., 80.]],
            "jptshift": [[14.], [80.], [45., 121.]],
            "jphi":     [[-0.3], [0.8], [-1.1, 2.5]],
            "evvars": {
                "MetUnclustEnUpDeltaX": [10., 15., 25.],
                "MetUnclustEnUpDeltaY": [5., 7., 13.],
            },
        }, {
            "metshift": [212.682763563916, 277.334010882431, 425.843625873921],
            "mephishift": [0.30325459685886, 0.561628679459254, 0.744090959514257],
        }],
    )
)
def test_objfunc_metshift(module, event, inputs, outputs):
    event.Electron_pt = awk.JaggedArray.fromiter([[],[],[]]).astype(np.float32)
    event.Electron_eta = awk.JaggedArray.fromiter([[],[],[]]).astype(np.float32)
    event.Muon_pt = awk.JaggedArray.fromiter([[],[],[]]).astype(np.float32)
    event.Tau_pt = awk.JaggedArray.fromiter([[],[],[]]).astype(np.float32)
    event.Photon_pt = awk.JaggedArray.fromiter([[],[],[]]).astype(np.float32)
    event.Photon_eta = awk.JaggedArray.fromiter([[],[],[]]).astype(np.float32)

    def sele(ev, source, nsig, attr):
        return awk.JaggedArray.fromiter([[],[],[]]).astype(np.float32)
    event.ElectronSelection = mock.Mock(side_effect=sele)
    event.MuonSelection = mock.Mock(side_effect=sele)
    event.TauSelection = mock.Mock(side_effect=sele)
    event.PhotonSelection = mock.Mock(side_effect=sele)

    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    met = np.array(inputs["met"], dtype=np.float32)
    mephi = np.array(inputs["mephi"], dtype=np.float32)
    event.MET_pt = met
    event.MET.pt = met
    event.MET_phi = mephi
    event.MET.phi = mephi

    jpt = awk.JaggedArray.fromiter(inputs["jpt"]).astype(np.float32)
    jptshift = awk.JaggedArray.fromiter(inputs["jptshift"]).astype(np.float32)
    jphi = awk.JaggedArray.fromiter(inputs["jphi"]).astype(np.float32)
    event.Jet.pt = jpt
    event.Jet_pt = jpt
    event.Jet.phi = jphi
    event.Jet_phi = jphi

    for key, val in inputs["evvars"].items():
        setattr(event.MET, key, np.array(val, dtype=np.float32))
        setattr(event, "MET_"+key, np.array(val, dtype=np.float32))

    module.begin(event)
    event.Jet.ptShift = mock.Mock(side_effect=lambda ev, source, nsig: jptshift)
    event.Jet_ptShift = mock.Mock(side_effect=lambda ev, source, nsig: jptshift)
    metshift = event.MET_ptShift(event, event.source, event.nsig)
    mephishift = event.MET_phiShift(event, event.source, event.nsig)

    assert np.allclose(
        metshift,
        np.array(outputs["metshift"]),
        rtol=1e-6, equal_nan=True,
    )

    assert np.allclose(
        mephishift,
        np.array(outputs["mephishift"]),
        rtol=1e-6, equal_nan=True,
    )


@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "nsig":   0,
            "source": '',
            "callable": False,
            "pt":     [[20.], [30.], [40., 50.], [60., 70.]],
            "mask":   [[True], [False], [True, False], [True, True]],
            "xclean": [[True], [True], [True, True], [False, True]],
        }, {
            "pt_noxclean":     [[20.], [], [40.], [60., 70.]],
            "pt_mask":         [[20.], [], [40.], [70.]],
        }], [{
            "nsig":   1,
            "source": 'Var1',
            "callable": True,
            "pt":     [[20.], [30.], [40., 50.], [60., 70.]],
            "mask":   [[True], [False], [True, False], [True, True]],
            "xclean": [[True], [True], [True, True], [False, True]],
        }, {
            "pt_noxclean":     [[20.], [], [40.], [60., 70.]],
            "pt_mask":         [[20.], [], [40.], [70.]],
        }],
    )
)
def test_objfunc_xclean(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    pt = awk.JaggedArray.fromiter(inputs["pt"]).astype(np.float32)
    xclean_mask = awk.JaggedArray.fromiter(inputs["xclean"]).astype(np.bool8)
    mask = awk.JaggedArray.fromiter(inputs["mask"]).astype(np.bool8)

    def cpt(ev, source, nsig):
        return pt

    for objname, selection, xclean in selections:
        if inputs["callable"]:
            setattr(event, objname+"_pt", pt)
        else:
            setattr(event, objname+"_pt", cpt)
        setattr(
            event, "{}_{}Mask".format(objname, selection),
            mock.Mock(side_effect=lambda ev, source, nsig: mask),
        )
        if xclean:
            if hasattr(event, "{}_XCleanMask"):
                continue
            setattr(
                event, "{}_XCleanMask".format(objname),
                mock.Mock(side_effect=lambda ev, source, nsig: xclean_mask),
            )

    pt_noxclean = awk.JaggedArray.fromiter(
        outputs["pt_noxclean"],
    ).astype(np.float32)
    pt_mask = awk.JaggedArray.fromiter(
        outputs["pt_mask"],
    ).astype(np.float32)

    module.begin(event)
    for objname, selection, xclean in selections:
        out_mask = getattr(event, selection)(event, event.source, event.nsig, 'pt')
        if xclean:
            out_noxclean = getattr(event, selection+"NoXClean")(event, event.source, event.nsig, 'pt')
            assert np.array_equal(out_mask.starts, pt_mask.starts)
            assert np.array_equal(out_mask.stops, pt_mask.stops)
            assert np.array_equal(out_mask.content, pt_mask.content)
            assert np.array_equal(out_noxclean.starts, pt_noxclean.starts)
            assert np.array_equal(out_noxclean.stops, pt_noxclean.stops)
            assert np.array_equal(out_noxclean.content, pt_noxclean.content)
        else:
            assert np.array_equal(out_mask.starts, pt_noxclean.starts)
            assert np.array_equal(out_mask.stops, pt_noxclean.stops)
            assert np.array_equal(out_mask.content, pt_noxclean.content)

import pytest
import mock
import numpy as np
import awkward as awk

from zinv.sequence.Readers import EventFunctions

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.nsig = 0
        self.source = ''
        self.cache = {}
        self.attribute_variation_sources = ["unclust"]

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return EventFunctions()

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "met":   [95., 203.],
            "mephi": [0.3, 0.5],
            "mupt":  [[40.], [201.]],
            "muphi": [[-0.2], [0.9]],
            "ept":   [[], [26., 91.]],
            "ephi":  [[], [-1.5, -1.3]],
            "source": "",
            "nsig": 0.,
        }, {
            "metnox":   [131.5090395, 358.2540117],
            "mephinox": [0.1536553484, 0.4049836823],
        }], [{
            "met":   [95., 203.],
            "mephi": [0.3, 0.5],
            "mupt":  [[40.], [201.]],
            "muphi": [[-0.2], [0.9]],
            "ept":   [[], [26., 91.]],
            "ephi":  [[], [-1.5, -1.3]],
            "source": "unclust",
            "nsig": 1.,
        }, {
            "metnox":   [131.5090395, 358.2540117],
            "mephinox": [0.1536553484, 0.4049836823],
        }],
    )
)
def test_metnox(event, module, inputs, outputs):
    event.source = inputs["source"]
    event.nsig = inputs["nsig"]
    module.begin(event)

    def met_ptshift(self):
        return np.array(inputs["met"], dtype=np.float32)
    event.MET_ptShift = mock.Mock(side_effect=met_ptshift)

    def met_phishift(self):
        return np.array(inputs["mephi"], dtype=np.float32)
    event.MET_phiShift = mock.Mock(side_effect=met_phishift)

    def muon_selection(self, attr):
        if attr == 'ptShift':
            return awk.JaggedArray.fromiter(inputs["mupt"]).astype(np.float32)
        elif attr == 'phi':
            return awk.JaggedArray.fromiter(inputs["muphi"]).astype(np.float32)
        else:
            print(attr)
            assert False
    event.MuonSelection = mock.Mock(side_effect=muon_selection)

    def ele_selection(self, attr):
        if attr == 'ptShift':
            return awk.JaggedArray.fromiter(inputs["ept"]).astype(np.float32)
        elif attr == 'phi':
            return awk.JaggedArray.fromiter(inputs["ephi"]).astype(np.float32)
        else:
            print(attr)
            assert False
    event.ElectronSelection = mock.Mock(side_effect=ele_selection)

    metnox_pt = event.METnoX_pt(event)
    metnox_phi = event.METnoX_phi(event)

    # Check arrays match with 0.0001% (1e-6)
    assert np.allclose(
        metnox_pt,
        np.array(outputs["metnox"], dtype=np.float32),
        rtol = 1e-6,
    )
    assert np.allclose(
        metnox_phi,
        np.array(outputs["mephinox"], dtype=np.float32),
        rtol = 1e-6,
    )

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "mephinox": [0.0, 0.5, 0.9, 1.3, 1.7, 2.1],
            "jphi": [
                [],
                [0.0],
                [0.0, 0.5],
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0, 1.5],
                [0.5, 1.0, 1.5, 2.0, 2.5],
            ],
            "source": "",
            "nsig": 0.,
        }, {
            "mindphi": [np.nan, 0.5, 0.4, 0.3, 0.2, 0.1],
        }], [{
            "mephinox": [0.0, 0.5, 0.9, 1.3, 1.7, 2.1],
            "jphi": [
                [],
                [0.0],
                [0.0, 0.5],
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0, 1.5],
                [0.5, 1.0, 1.5, 2.0, 2.5],
            ],
            "source": "unclust",
            "nsig": 1.,
        }, {
            "mindphi": [np.nan, 0.5, 0.4, 0.3, 0.2, 0.1],
        }],
    )
)
def test_mindphi(event, module, inputs, outputs):
    event.source = inputs["source"]
    event.nsig = inputs["nsig"]
    module.begin(event)

    event.METnoX_phi = mock.Mock(
        side_effect=lambda ev: np.array(inputs["mephinox"], dtype=np.float32),
    )

    def jet_selection(self, attr):
        assert attr == 'phi'
        return awk.JaggedArray.fromiter(inputs["jphi"]).astype(np.float32)
    event.JetSelection = mock.Mock(side_effect=jet_selection)

    assert np.allclose(
        event.MinDPhiJ1234METnoX(event),
        np.array(outputs["mindphi"], dtype=np.float32),
        rtol = 1e-6, equal_nan=True,
    )

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "cmet":   [50., 75., 150., 150.],
            "met":    [100., 200., 150., 150.],
            "metnox": [250., 0., 200., 0.],
            "source": "",
            "nsig": 0.,
        }, {
            "dcalomet": [0.2, np.inf, 0., np.nan],
        }], [{
            "cmet":   [50., 75., 150., 150.],
            "met":    [100., 200., 150., 150.],
            "metnox": [250., 0., 200., 0.],
            "source": "unclust",
            "nsig": 1.,
        }, {
            "dcalomet": [0.2, np.inf, 0., np.nan],
        }],
    )
)
def test_met_dcalo(event, module, inputs, outputs):
    event.source = inputs["source"]
    event.nsig = inputs["nsig"]

    module.begin(event)

    event.CaloMET_pt = np.array(inputs["cmet"], dtype=np.float32)
    event.MET_ptShift = mock.Mock(
        side_effect=lambda ev: np.array(inputs["met"], dtype=np.float32),
    )
    event.METnoX_pt = mock.Mock(
        side_effect=lambda ev: np.array(inputs["metnox"], dtype=np.float32),
    )

    assert np.allclose(
        event.MET_dCaloMET(event),
        np.array(outputs["dcalomet"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "met":   [100., 200., 300., 400., 500.],
            "mephi": [-1., 0.5, 0., 0.5, 1.],
            "mupt":  [[], [50.], [60., 70.], [], [80.]],
            "muphi": [[], [0.1], [-0.1, 0.3], [], [-0.5]],
            "ept":   [[], [], [], [40.], [45.]],
            "ephi":  [[], [], [], [2.1], [-2.1]],
            "source": "",
            "nsig": 0.,
        }, {
            "mtw": [np.nan, 39.73386616, np.nan, 181.4783313, np.nan],
        }], [{
            "met":   [100., 200., 300., 400., 500.],
            "mephi": [-1., 0.5, 0., 0.5, 1.],
            "mupt":  [[], [50.], [60., 70.], [], [80.]],
            "muphi": [[], [0.1], [-0.1, 0.3], [], [-0.5]],
            "ept":   [[], [], [], [40.], [45.]],
            "ephi":  [[], [], [], [2.1], [-2.1]],
            "source": "unclust",
            "nsig": 1.,
        }, {
            "mtw": [np.nan, 39.73386616, np.nan, 181.4783313, np.nan],
        }],
    )
)
def test_mtw(event, module, inputs, outputs):
    event.source = inputs["source"]
    event.nsig = inputs["nsig"]

    module.begin(event)

    def muon_selection(self, attr):
        if attr == "ptShift":
            return awk.JaggedArray.fromiter(inputs["mupt"]).astype(np.float32)
        elif attr == "phi":
            return awk.JaggedArray.fromiter(inputs["muphi"]).astype(np.float32)
        else:
            print(attr)
            assert False
    def ele_selection(self, attr):
        if attr == "ptShift":
            return awk.JaggedArray.fromiter(inputs["ept"]).astype(np.float32)
        elif attr == "phi":
            return awk.JaggedArray.fromiter(inputs["ephi"]).astype(np.float32)
        else:
            print(attr)
            assert False

        return awk.JaggedArray([0, 0, 0, 0, 1], [0, 0, 0, 1, 2], content)

    event.MET_ptShift = mock.Mock(
        side_effect = lambda ev: np.array(inputs["met"], dtype=np.float32),
    )
    event.MET_phiShift = mock.Mock(
        side_effect = lambda ev: np.array(inputs["mephi"], dtype=np.float32),
    )
    event.MuonSelection = mock.Mock(side_effect=muon_selection)
    event.ElectronSelection = mock.Mock(side_effect=ele_selection)

    assert np.allclose(
        event.MTW(event),
        np.array(outputs["mtw"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "mupt":   [[], [50., 60.], [], [70., 80.]],
            "mueta":  [[], [-2.1, 1.4], [], [-0.5, 0.6]],
            "muphi":  [[], [0.1, -0.1], [], [0.3, -0.5]],
            "mumass": [[], [0., 0.], [], [0., 0.]],
            "elpt":   [[], [], [40., 45.], [50., 55.]],
            "eleta":  [[], [], [0.4, 0.8], [1.2, 1.6]],
            "elphi":  [[], [], [2.1, -2.1], [0.2, -0.2]],
            "elmass": [[], [], [0., 0.], [0., 0.]],
            "source": "",
            "nsig": 0.,
        }, {
            "mll": [np.nan, 305.8701498, 75.21169786, np.nan],
        }], [{
            "mupt":   [[], [50., 60.], [], [70., 80.]],
            "mueta":  [[], [-2.1, 1.4], [], [-0.5, 0.6]],
            "muphi":  [[], [0.1, -0.1], [], [0.3, -0.5]],
            "mumass": [[], [0., 0.], [], [0., 0.]],
            "elpt":   [[], [], [40., 45.], [50., 55.]],
            "eleta":  [[], [], [0.4, 0.8], [1.2, 1.6]],
            "elphi":  [[], [], [2.1, -2.1], [0.2, -0.2]],
            "elmass": [[], [], [0., 0.], [0., 0.]],
            "source": "unclust",
            "nsig": 1.,
        }, {
            "mll": [np.nan, 305.8701498, 75.21169786, np.nan],
        }],
    )
)
def test_mll(event, module, inputs, outputs):
    event.size = len(inputs["mupt"])
    event.source = inputs["source"]
    event.nsig = inputs["nsig"]

    module.begin(event)

    def muon_selection(self, attr):
        if attr == "ptShift":
            return awk.JaggedArray.fromiter(inputs["mupt"]).astype(np.float32)
        elif attr == "eta":
            return awk.JaggedArray.fromiter(inputs["mueta"]).astype(np.float32)
        elif attr == "phi":
            return awk.JaggedArray.fromiter(inputs["muphi"]).astype(np.float32)
        elif attr == "mass":
            return awk.JaggedArray.fromiter(inputs["mumass"]).astype(np.float32)
        else:
            print(attr)
            assert False
    def ele_selection(self, attr):
        if attr == "ptShift":
            return awk.JaggedArray.fromiter(inputs["elpt"]).astype(np.float32)
        elif attr == "eta":
            return awk.JaggedArray.fromiter(inputs["eleta"]).astype(np.float32)
        elif attr == "phi":
            return awk.JaggedArray.fromiter(inputs["elphi"]).astype(np.float32)
        elif attr == "mass":
            return awk.JaggedArray.fromiter(inputs["elmass"]).astype(np.float32)
        else:
            print(attr)
            assert False

    event.MuonSelection = mock.Mock(side_effect=muon_selection)
    event.ElectronSelection = mock.Mock(side_effect=ele_selection)

    assert np.allclose(
        event.MLL(event),
        np.array(outputs["mll"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "mucharge": [[], [1], [], [-1]],
            "elcharge": [[], [], [-1], [1]],
            "source": "",
            "nsig": 0.,
        }, {
            "lepcharge": [0, 1, -1, 0],
        }], [{
            "mucharge": [[], [1], [], [-1]],
            "elcharge": [[], [], [-1], [1]],
            "source": "unclust",
            "nsig": 1.,
        }, {
            "lepcharge": [0, 1, -1, 0],
        }],
    )
)
def test_lepton_charge(event, module, inputs, outputs):
    event.size = len(inputs["mucharge"])
    event.source = inputs["source"]
    event.nsig = inputs["nsig"]

    module.begin(event)

    def muon_selection(self, attr):
        assert attr == 'charge'
        return awk.JaggedArray.fromiter(inputs["mucharge"]).astype(np.int32)
    def ele_selection(self, attr):
        assert attr == 'charge'
        return awk.JaggedArray.fromiter(inputs["elcharge"]).astype(np.int32)
    event.MuonSelection = mock.Mock(side_effect=muon_selection)
    event.ElectronSelection = mock.Mock(side_effect=ele_selection)

    assert np.allclose(
        event.LeptonCharge(event),
        np.array(outputs["lepcharge"], dtype=np.int32),
        rtol=1e-6, equal_nan=True,
    )

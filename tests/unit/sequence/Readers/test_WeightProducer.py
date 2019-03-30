import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.sequence.Readers import WeightProducer

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.cache = {}
        self.config = mock.MagicMock()

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return WeightProducer(
        weight_sequence_path = "dummy_path.yaml",
        nuisances = None,
    )

data = """
weights:
    MET:
        Monojet:
            Regions:
                - "Monojet"
            Data:
                - "ev: np.zeros(ev.size, dtype=np.float32)"
                - "ev: ev.IsMETTriggered"
            MC:
                - "ev: ev.MonojetMC_Weights"
        Muon:
            Regions:
                - "SingleMuon"
                - "SingleMuonQCD"
            Data:
                - "ev: ev.IsMETTriggered"
            MC:
                - "ev: ev.SingleMuonMC_Weights"
    SingleElectron:
        Electron:
            Regions:
                - "SingleElectron"
                - "SingleElectronQCD"
            Data:
                - "ev: ev.IsSingleElectronTriggered"
            MC:
                - "ev: ev.SingleElectronMC1_Weights"
                - "ev: ev.SingleElectronMC2_Weights"
                - "ev: ev.SingleElectronMC3_Weights"
weight_variations:
    - "metTrigStat"
    - "muonIdTightSF"
attribute_variations:
    - "jesTotal"
    - "eleEnergyScale"
"""

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "isdata": True,
            "nuisances": None,
        }, {
            "wvars": ["metTrigStat", "muonIdTightSF"],
            "avars": ["jesTotal", "eleEnergyScale"],
        }], [{
            "isdata": True,
            "nuisances": ["metTrigStat", "jesTotal"],
        }, {
            "wvars": ["metTrigStat"],
            "avars": ["jesTotal"],
        }],
    )
)
def test_weightproducer_begin(module, event, inputs, outputs):
    event.config.dataset.isdata = inputs["isdata"]
    module.nuisances = inputs["nuisances"]

    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)

    assert event.weight_variation_sources == outputs["wvars"]
    assert event.attribute_variation_sources == outputs["avars"]
    assert module.weight_sequence_path == "dummy_path.yaml"

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
#            "isdata": True,
#            "wtuple": [
#                ("MET",            "Monojet",           "Data"),
#                ("MET",            "SingleMuon",        "Data"),
#                ("MET",            "SingleMuonQCD",     "Data"),
#                ("SingleElectron", "SingleElectron",    "Data"),
#                ("SingleElectron", "SingleElectronQCD", "Data"),
#            ],
#            "evattrs": [
#                ("IsMETTriggered",            [True, True,  False]),
#                ("IsSingleElectronTriggered", [True, False, False]),
#            ],
#            "nuisances": None,
#        }, {
#            "weights": [
#                ("MET_Monojet_Data",                      [0., 0., 0.]),
#                ("MET_SingleMuon_Data",                   [1., 1., 0.]),
#                ("MET_SingleMuonQCD_Data",                [1., 1., 0.]),
#                ("SingleElectron_SingleElectron_Data",    [1., 0., 0.]),
#                ("SingleElectron_SingleElectronQCD_Data", [1., 0., 0.]),
#            ]
#        }], [{
            "isdata": False,
            "wtuple": [
                ("MET",            "Monojet",           "MC"),
                ("MET",            "SingleMuon",        "MC"),
                ("MET",            "SingleMuonQCD",     "MC"),
                ("SingleElectron", "SingleElectron",    "MC"),
                ("SingleElectron", "SingleElectronQCD", "MC"),
            ],
            "evattrs": [
                ("MonojetMC_Weights",         [0.3, 0.6, 0.9]),
                ("SingleMuonMC_Weights",      [1.5, 1.1, 0.5]),
                ("SingleElectronMC1_Weights", [0.9, 0.4, 1.2]),
                ("SingleElectronMC2_Weights", [0.1, 0.3, 0.5]),
                ("SingleElectronMC3_Weights", [2.4, 3.2, 0.1]),
            ],
            "nuisances": None,
        }, {
            "weights": [
                ("MET_Monojet_MC",                      [0.3,   0.6,   0.9]),
                ("MET_SingleMuon_MC",                   [1.5,   1.1,   0.5]),
                ("MET_SingleMuonQCD_MC",                [1.5,   1.1,   0.5]),
                ("SingleElectron_SingleElectron_MC",    [0.216, 0.384, 0.06]),
                ("SingleElectron_SingleElectronQCD_MC", [0.216, 0.384, 0.06]),
            ]
        }],
    )
)
def test_weightproducer_weights(module, event, inputs, outputs):
    event.size = len(inputs["evattrs"][0][1])
    event.config.dataset.isdata = inputs["isdata"]
    module.nuisances = inputs["nuisances"]

    for attr, arr in inputs["evattrs"]:
        dtype = np.float32
        if inputs["isdata"]:
            dtype = np.bool8
        setattr(event, attr, np.array(arr, dtype=dtype))

    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)

    for label, arr in outputs["weights"]:
        win = getattr(event, "Weight_{}".format(label))(event)
        wout = np.array(arr, dtype=np.float32)
        print(label)
        print(win)
        print(wout)
        assert np.allclose(win, wout, rtol=1e-6, equal_nan=True)

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "isdata": True,
            "nuisances": None,
        }, {
            "wvars": ["metTrigStat", "muonIdTightSF"],
            "avars": ["jesTotal", "eleEnergyScale"],
        }], [{
            "isdata": True,
            "nuisances": ["metTrigStat", "jesTotal"],
        }, {
            "wvars": ["metTrigStat"],
            "avars": ["jesTotal"],
        }],
    )
)
def test_weightproducer_event(module, event, inputs, outputs):
    event.config.dataset.isdata = inputs["isdata"]
    module.nuisances = inputs["nuisances"]

    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)
        event.weight_variation_sources = None
        event.attribute_variation_sources = None
        module.event(event)

    assert event.weight_variation_sources == outputs["wvars"]
    assert event.attribute_variation_sources == outputs["avars"]

def test_weightproducer_end(module):
    assert module.end() is None
    assert module.lambda_functions is None

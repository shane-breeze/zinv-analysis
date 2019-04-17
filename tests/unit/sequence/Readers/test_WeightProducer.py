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
            "wvars": ["metTrigStat", "muonIdTightSF"],
            "avars": ["jesTotal", "eleEnergyScale"],
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
            "isdata": True,
            "nuisances": None,
        }, {
            "wvars": ["metTrigStat", "muonIdTightSF"],
            "avars": ["jesTotal", "eleEnergyScale"],
        }], [{
            "isdata": True,
            "nuisances": ["metTrigStat", "jesTotal"],
        }, {
            "wvars": ["metTrigStat", "muonIdTightSF"],
            "avars": ["jesTotal", "eleEnergyScale"],
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

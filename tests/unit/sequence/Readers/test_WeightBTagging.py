import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.sequence.Readers import WeightBTagging

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.attribute_variation_sources = []
        self.cache = {}

        self.Jet = DummyColl()

@pytest.fixture()
def path():
    toppath = os.path.abspath(os.environ["TOPDIR"])
    datapath = os.path.join(toppath, "zinv/data")
    return datapath

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module(path):
    return WeightBTagging(
        operating_point = "medium",
        threshold = 0.8484,
        measurement_types = {"b": "comb", "c": "comb", "udsg": "incl"},
        calibration_file = path+"/btagging/CSVv2_Moriond17_B_H_params.csv",
    )

def test_weightbtagging_init(module, path):
    assert module.ops == {"loose": 0, "medium": 1, "tight": 2, "reshaping": 3}
    assert module.flavours == {"b": 0, "c": 1, "udsg": 2}
    assert module.hadron_to_flavour == {
        5: 0, -5: 0,
        4: 1, -4: 1,
        0: 2, 1: 2, 2: 2, 3: 2, -1: 2, -2: 2, -3: 2, 21: 2,
    }
    assert module.operating_point == "medium"
    assert module.threshold == 0.8484
    assert module.measurement_types == {"b": "comb", "c": "comb", "udsg": "incl"}
    assert module.calibration_file == path+"/btagging/CSVv2_Moriond17_B_H_params.csv"
    assert hasattr(module, 'calibrations')

def test_weightbtagging_end(module):
    assert module.end() is None
    assert module.calibrations is None

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "jflav": [[0], [1, 2]],
            "jeta": [[-2.3], [0.5, 3.1]],
            "jptshift": [[20.], [2000., 100.]],
        }, {
        }],
    )
)
def test_weightbtagging_begin(module, event, inputs, outputs):
    jflav = awk.JaggedArray.fromiter(inputs["jflav"]).astype(np.float32)
    jeta = awk.JaggedArray.fromiter(inputs["jeta"]).astype(np.float32)
    jptshift = awk.JaggedArray.fromiter(inputs["jptshift"]).astype(np.float32)
    event.Jet.hadronFlavour = jflav
    event.Jet_hadronFlavour = jflav
    event.Jet.eta = jeta
    event.Jet_eta = jeta
    event.Jet.ptShift = mock.Mock(side_effect=lambda ev: jptshift)
    event.Jet_ptShift = mock.Mock(side_effect=lambda ev: jptshift)

    assert module.begin(event) is None
    print(event.Jet_btagSF(event))
    #assert False

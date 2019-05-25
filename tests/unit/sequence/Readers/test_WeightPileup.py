import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.sequence.Readers import WeightPileup

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.attribute_variation_sources = []
        self.cache = {}

    def register_function(self, event, name, function):
        self.__dict__[name] = function

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def path():
    toppath = os.path.abspath(os.environ["TOPDIR"])
    datapath = os.path.join(toppath, "zinv/data")
    return datapath

@pytest.fixture()
def module(path):
    return WeightPileup(
        correction_file = path + "/pileup/nTrueInt_corrections.txt",
        variable = "Pileup_nTrueInt",
    )

def test_weightpileup_init(module, path):
    assert module.correction_file == path + "/pileup/nTrueInt_corrections.txt"
    assert module.variable == "Pileup_nTrueInt"

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "nTrueInt": [-10, 0, 1, 5, 10, 50, 100, 1000],
            "source": "",
            "nsig": 0.,
        }, {
            "wpu": [0.362381070783, 0.362381070783, 0.900423097032, 1.18999617534, 0.973981892207, 0.00443414269021, 0.00199174119977, 0.00199174119977],
        }], [{
            "nTrueInt": [-10, 0, 1, 5, 10, 50, 100, 1000],
            "source": "pileup",
            "nsig": 1.,
        }, {
            "wpu": [0.375449447985, 0.375449447985, 1.14948701696, 1.3084971864, 1.50130446585, 0.000650594369215, 0.00111524301666, 0.00111524301666],
        }], [{
            "nTrueInt": [-10, 0, 1, 5, 10, 50, 100, 1000],
            "source": "pileup",
            "nsig": -1.,
        }, {
            "wpu": [0.353102925676, 0.353102925676, 0.70898051882, 1.07170878202, 0.638579609949, 0.0219738931573, 0.0031258061505, 0.0031258061505],
        }], [{
            "nTrueInt": [-10, 0, 1, 5, 10, 50, 100, 1000],
            "source": "other",
            "nsig": -1.,
        }, {
            "wpu": [0.362381070783, 0.362381070783, 0.900423097032, 1.18999617534, 0.973981892207, 0.00443414269021, 0.00199174119977, 0.00199174119977],
        }],
    )
)
def test_weightpileup_begin(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]
    module.begin(event)
    event.Pileup_nTrueInt = np.array(inputs["nTrueInt"], dtype=np.float32)
    wpu = event.WeightPU(event, event.source, event.nsig)
    owpu = np.array(outputs["wpu"], dtype=np.float32)

    print(wpu)
    print(owpu)
    assert np.allclose(wpu, owpu, rtol=1e-6, equal_nan=True)

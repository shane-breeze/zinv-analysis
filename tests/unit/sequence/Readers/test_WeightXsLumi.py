import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.sequence.Readers import WeightXsLumi

class DummyColl(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.cache = {}
        self.config = mock.MagicMock()

    def register_function(self, event, name, function):
        self.__dict__[name] = function

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return WeightXsLumi()

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "xsection": 10.,
            "lumi": 3.3,
            "associate_weights": [2., 5., 10.],
            "genWeight": [0.2, 0.4, 0.6, 0.8, 1.0],
        }, {
            "weight": [3.88235294118E-01, 7.76470588235E-01, 1.16470588235E+00, 1.55294117647E+00, 1.94117647059E+00],
        }],
    )
)
def test_weightxslumi_begin(module, event, inputs, outputs):
    associates = []
    for w in inputs["associate_weights"]:
        associates.append(DummyColl(
            sumweights = w,
        ))
    event.config.dataset.associates = associates
    event.config.dataset.xsection = inputs["xsection"]
    event.config.dataset.lumi = inputs["lumi"]
    event.genWeight = np.array(inputs["genWeight"], dtype=np.float32)

    module.begin(event)

    weight = event.WeightXsLumi(event)
    oweight = np.array(outputs["weight"], dtype=np.float32)
    assert np.allclose(weight, oweight, rtol=1e-6, equal_nan=True)

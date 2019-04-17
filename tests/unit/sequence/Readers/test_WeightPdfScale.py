import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.sequence.Readers import WeightPdfScale

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.attribute_variation_sources = []
        self.cache = {}

        self.config = mock.MagicMock()

    def delete_branches(self, branches):
        pass

    def hasbranch(self, branch):
        return True

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return WeightPdfScale(
        parents_to_skip = ["QCD", "SingleTop"],
    )

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "parent": "ZJetsToNuNu",
            "origpdf": [0., 1.02, 1.01],
            "npdf": 5,
            "pdf": [
                [0.65, 0.94, 0.91, 1.90, 0.50],
                [1.01, 1.02, 1.03, 1.04, 1.05],
                [0.99, 1.01, 0.98, 1.02, 1.1]
            ],
            "nscale": 9,
            "scale": [
                [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
                [0.91, 1.01, 0.92, 1.02, 0.93, 1.03, 0.94, 1.04, 0.95],
                [0.99, 1.01, 0.98, 1.02, 0.97, 1.03, 0.96, 1.04, 0.95],
            ],
            "source": "",
            "nsig": 0.,
        }, {
            "pdf": [0.91, 1.03, 0.98],
            "scale": [1.04, 1.02, 1.02],
        }], [{
            "parent": "ZJetsToNuNu",
            "origpdf": [0., 1.02, 1.01],
            "npdf": 5,
            "pdf": [
                [0.65, 0.94, 0.91, 1.90, 0.50],
                [1.01, 1.02, 1.03, 1.04, 1.05],
                [0.99, 1.01, 0.98, 1.02, 1.1]
            ],
            "nscale": 9,
            "scale": [
                [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
                [0.91, 1.01, 0.92, 1.02, 0.93, 1.03, 0.94, 1.04, 0.95],
                [0.99, 1.01, 0.98, 1.02, 0.97, 1.03, 0.96, 1.04, 0.95],
            ],
            "source": "pdf1",
            "nsig": 1.,
        }, {
            "pdf": [0.91, 1.03, 0.98],
            "scale": [1.04, 1.02, 1.02],
        }], [{
            "parent": "ZJetsToNuNu",
            "origpdf": [0., 1.02, 1.01],
            "npdf": 5,
            "pdf": [
                [0.65, 0.94, 0.91, 1.90, 0.50],
                [1.01, 1.02, 1.03, 1.04, 1.05],
                [0.99, 1.01, 0.98, 1.02, 1.1]
            ],
            "nscale": 9,
            "scale": [
                [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
                [0.91, 1.01, 0.92, 1.02, 0.93, 1.03, 0.94, 1.04, 0.95],
                [0.99, 1.01, 0.98, 1.02, 0.97, 1.03, 0.96, 1.04, 0.95],
            ],
            "source": "muf_scale",
            "nsig": -1.,
        }, {
            "pdf": [0.91, 1.03, 0.98],
            "scale": [1.04, 1.02, 1.02],
        }], [{
            "parent": "QCD",
            "origpdf": [0., 1.02, 1.01],
            "npdf": 5,
            "pdf": [
                [0.65, 0.94, 0.91, 1.90, 0.50],
                [1.01, 1.02, 1.03, 1.04, 1.05],
                [0.99, 1.01, 0.98, 1.02, 1.1]
            ],
            "nscale": 9,
            "scale": [
                [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
                [0.91, 1.01, 0.92, 1.02, 0.93, 1.03, 0.94, 1.04, 0.95],
                [0.99, 1.01, 0.98, 1.02, 0.97, 1.03, 0.96, 1.04, 0.95],
            ],
            "source": "muf_scale",
            "nsig": 1.,
        }, {
            "pdf": [0.91, 1.03, 0.98],
            "scale": [1.04, 1.02, 1.02],
        }], [{
            "parent": "SingleTop",
            "origpdf": [0., 1.02, 1.01],
            "npdf": 5,
            "pdf": [
                [0.65, 0.94, 0.91, 1.90, 0.50],
                [1.01, 1.02, 1.03, 1.04, 1.05],
                [0.99, 1.01, 0.98, 1.02, 1.1]
            ],
            "nscale": 9,
            "scale": [
                [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09],
                [0.91, 1.01, 0.92, 1.02, 0.93, 1.03, 0.94, 1.04, 0.95],
                [0.99, 1.01, 0.98, 1.02, 0.97, 1.03, 0.96, 1.04, 0.95],
            ],
            "source": "pdf2",
            "nsig": -1.,
        }, {
            "pdf": [0.91, 1.03, 0.98],
            "scale": [1.04, 1.02, 1.02],
        }],
    )
)
def test_weightpdfscale(module, event, inputs, outputs):
    event.source = inputs["source"]
    event.nsig = inputs["nsig"]
    event.config.dataset.parent = inputs["parent"]
    event.size = len(inputs["origpdf"])

    module.begin(event)

    iorigpdf = np.array(inputs["origpdf"], dtype=np.float32)
    ipdf = awk.JaggedArray.fromiter(inputs["pdf"]).astype(np.float32)
    iscale = awk.JaggedArray.fromiter(inputs["scale"]).astype(np.float32)

    event.LHEWeight_originalXWGTUP = iorigpdf
    event.nLHEPdfWeight = inputs["npdf"]
    event.nLHEScaleWeight = inputs["nscale"]
    event.LHEPdfWeight = ipdf
    event.LHEScaleWeight = iscale

    pdf = event.WeightPdfVariations(event, 2)
    scale = event.WeightScaleVariations(event, 3)
    opdf = np.array(outputs["pdf"], dtype=np.float32)
    oscale = np.array(outputs["scale"], dtype=np.float32)

    print(pdf)
    print(opdf)

    print(scale)
    print(oscale)

    assert np.allclose(pdf, opdf, rtol=1e-6, equal_nan=True)
    assert np.allclose(scale, oscale, rtol=1e-6, equal_nan=True)

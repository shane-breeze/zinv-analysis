import pytest
import mock
import numpy as np
import awkward as awk

from zinv.sequence.Readers import ObjectCrossCleaning

class DummyColl(object):
    def __init__(self):
        pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.nsig = 0
        self.source = ''
        self.attribute_variation_sources = ["variation"]

        self.C1 = DummyColl()
        self.RC1 = DummyColl()
        self.RC2 = DummyColl()
        self.cache = {}

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return ObjectCrossCleaning(
        collections = ("C1",),
        ref_collections = ("RC1", "RC2"),
        mindr = 0.4,
    )

def test_objxclean_init(module):
    assert module.collections == ("C1",)
    assert module.ref_collections == ("RC1", "RC2")
    assert module.mindr == 0.4

def test_objxclean_begin(module, event):
    assert module.begin(event) is None
    assert hasattr(event, "C1_XCleanMask")

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "c1_eta": [[0.3], [0.9, 1.2], [0.6, 1.5, 1.8]],
            "c1_phi": [[0.15], [0.75, 1.05], [0.45, 1.35, 1.65]],
            "rc1_eta": [[], [1.1], [2.5]],
            "rc1_phi": [[], [0.78], [1.65]],
            "rc2_eta": [[], [], [1.45]],
            "rc2_phi": [[], [], [1.38]],
            "source": "",
            "nsig": 0.,
        }, {
            "xclean": [[True], [False, False], [True, False, True]],
        }], [{
            "c1_eta": [[0.3], [0.9, 1.2], [0.6, 1.5, 1.8]],
            "c1_phi": [[0.15], [0.75, 1.05], [0.45, 1.35, 1.65]],
            "rc1_eta": [[], [1.1], [2.5]],
            "rc1_phi": [[], [0.78], [1.65]],
            "rc2_eta": [[], [], [1.45]],
            "rc2_phi": [[], [], [1.38]],
            "source": "variation",
            "nsig": 1.,
        }, {
            "xclean": [[True], [False, False], [True, False, True]],
        }],
    )
)
def test_objxclean_attr(module, event, inputs, outputs):
    event.C1.eta = awk.JaggedArray.fromiter(inputs["c1_eta"]).astype(np.float32)
    event.C1.phi = awk.JaggedArray.fromiter(inputs["c1_phi"]).astype(np.float32)

    def rc1_call(ev, attr):
        if attr == "eta":
            return awk.JaggedArray.fromiter(
                inputs["rc1_eta"],
            ).astype(np.float32)
        elif attr == "phi":
            return awk.JaggedArray.fromiter(
                inputs["rc1_phi"],
            ).astype(np.float32)
        else:
            print(attr)
            assert False
    event.RC1 = mock.Mock(side_effect=rc1_call)

    def rc2_call(ev, attr):
        if attr == "eta":
            return awk.JaggedArray.fromiter(
                inputs["rc2_eta"],
            ).astype(np.float32)
        elif attr == "phi":
            return awk.JaggedArray.fromiter(
                inputs["rc2_phi"],
            ).astype(np.float32)
        else:
            print(attr)
            assert False
    event.RC2 = mock.Mock(side_effect=rc2_call)

    module.begin(event)
    xclean = event.C1_XCleanMask(event)
    oxclean = awk.JaggedArray.fromiter(outputs["xclean"]).astype(np.bool8)

    assert np.array_equal(xclean.starts, oxclean.starts)
    assert np.array_equal(xclean.stops, oxclean.stops)
    assert np.array_equal(xclean.content, oxclean.content)

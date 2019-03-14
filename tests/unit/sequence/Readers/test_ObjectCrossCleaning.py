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
        self.attribute_variation_sources = []

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

def test_objxclean_attr(module, event):
    c1_starts = [0, 1, 3]
    c1_stopys = [1, 3, 6]
    event.C1.eta = awk.JaggedArray(
        c1_starts, c1_stopys, [0.3, 0.9, 1.2, 0.6 ,1.5, 1.8],
    )
    event.C1.phi = awk.JaggedArray(
        c1_starts, c1_stopys, [0.15, 0.75, 1.05, 0.45, 1.35, 1.65],
    )

    rc1_starts = [0, 0, 1]
    rc1_stopys = [0, 1, 2]
    def rc1_call(ev, attr):
        if attr == "eta":
            content = [1.1, 2.5]
        elif attr == "phi":
            content = [0.78, 1.65]
        else:
            assert False
        return awk.JaggedArray(rc1_starts, rc1_stopys, content)
    event.RC1 = mock.Mock(side_effect=rc1_call)

    rc2_starts = [0, 0, 0]
    rc2_stopys = [0, 0, 1]
    def rc2_call(ev, attr):
        if attr == "eta":
            content = [1.45]
        elif attr == "phi":
            content = [1.38]
        else:
            assert False
        return awk.JaggedArray(rc2_starts, rc2_stopys, content)
    event.RC2 = mock.Mock(side_effect=rc2_call)

    module.begin(event)
    xclean = event.C1_XCleanMask(event)

    assert np.array_equal(xclean.starts, np.array(c1_starts))
    assert np.array_equal(xclean.stops, np.array(c1_stopys))
    assert np.array_equal(
        xclean.content,
        np.array([True, False, False, True, False, True]),
    )

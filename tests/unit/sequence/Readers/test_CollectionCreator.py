import pytest
import mock
import numpy as np
import awkward as awk

from sequence.Readers import CollectionCreator, Collection

class DummyEvent(object):
    def __init__(self):
        self.MET_pt = np.array([10., 20., 30., 40., 50.])
        self.MET_phi = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.Jet_pt = awk.JaggedArray(
            [0, 1, 3, 4, 7],
            [1, 3, 4, 7, 8],
            [60., 70., 80., 90., 100.],
        )
        self.Jet_phi = awk.JaggedArray(
            [0, 1, 3, 4, 7],
            [1, 3, 4, 7, 8],
            [0.6, 0.7, 0.8, 0.9, 1.0],
        )

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return CollectionCreator(collections=["MET", "Jet"])

@pytest.fixture()
def collection(event, module):
    module.event(event)
    return event.Jet

def test_collection_creator(event, module):
    module.event(event)
    assert np.array_equal(event.MET.pt, event.MET_pt)
    assert np.array_equal(event.MET.phi, event.MET_phi)
    assert np.array_equal(event.Jet.pt.content, event.Jet_pt.content)
    assert np.array_equal(event.Jet.pt.starts, event.Jet_pt.starts)
    assert np.array_equal(event.Jet.pt.stops, event.Jet_pt.stops)
    assert np.array_equal(event.Jet.phi.content, event.Jet_phi.content)
    assert np.array_equal(event.Jet.phi.starts, event.Jet_phi.starts)
    assert np.array_equal(event.Jet.phi.stops, event.Jet_phi.stops)

def test_collection_init(collection, event):
    assert collection.name == "Jet"
    assert collection.event == event
    assert collection.ref_name == None

def test_collection_getattr(collection, event):
    assert np.array_equal(collection.pt.content, event.Jet_pt.content)
    assert np.array_equal(collection.pt.starts, event.Jet_pt.starts)
    assert np.array_equal(collection.pt.stops, event.Jet_pt.stops)

def test_collection_getattr_attrerror(collection):
    with pytest.raises(AttributeError, match="name should be defined but isn't"):
        del collection.name
        collection.name

def test_collection_getattr_evattrerror(collection):
    with pytest.raises(AttributeError, match="'DummyEvent' object has no attribute 'Jet_eta'"):
        collection.eta

def test_collection_repr(collection):
    assert repr(collection) == "Collection(name = 'Jet', ref_name = None)"

import pytest
import mock
import numpy as np
import awkward as awk

from zinv.sequence.Readers import SkimCollections

class DummyColl(object):
    def __init__(self):
        pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.

        self.C1 = DummyColl()

        self.attribute_variation_sources = ['']
        self.cache = {}

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return SkimCollections(physics_object_selection_path="dummy_path.yaml")

def test_skimcollections_init(module):
    assert module.physics_object_selection_path == "dummy_path.yaml"

def test_skimcollections_open(module, event):
    data = """C1Sub:\n    original: \"C1\"\n    selections:\n        - \"ev: ev.C1Selection(ev)\""""
    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)
    assert mocked_open.call_args_list == [mock.call("dummy_path.yaml", "r")]

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "dummy_selection": [[True], [False, True]],
        }, {
            "selection": [[True], [False, True]],
        }],
    )
)
def test_skimcollections_begin_onemask(module, event, inputs, outputs):
    collection_selection = awk.JaggedArray.fromiter(inputs["dummy_selection"])
    event.C1Selection = lambda ev: collection_selection
    event.C1.pt = collection_selection*100.

    data = """C1Sub:\n    original: \"C1\"\n    selections:\n        - \"ev: ev.C1Selection(ev)\""""
    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)

    mask = event.C1_C1SubMask(event)
    out_mask = awk.JaggedArray.fromiter(outputs["selection"])
    assert np.array_equal(mask.starts, out_mask.starts)
    assert np.array_equal(mask.stops, out_mask.stops)
    assert np.array_equal(mask.content, out_mask.content)

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "c1veto": [[True], [False, True]],
            "c1sele": [[False], [False, True]],
        }, {
            "mask_veto": [[True], [False, True]],
            "mask_sele": [[False], [False, True]],
            "mask_vetonosele": [[True], [False, False]],
        }],
    )
)
def test_skimcollections_begin_allmasks(module, event, inputs, outputs):
    collection_veto = awk.JaggedArray.fromiter(inputs["c1veto"])
    collection_selection = awk.JaggedArray.fromiter(inputs["c1sele"])
    event.C1Veto = lambda ev: collection_veto
    event.C1Selection = lambda ev: collection_selection
    event.C1.pt = collection_selection*100.

    data = """C1Veto:\n    original: \"C1\"\n    selections:\n        - \"ev: ev.C1Veto(ev)\"\n"""\
            + """C1Selection:\n    original: \"C1\"\n    selections:\n        - \"ev: ev.C1Selection(ev)\""""
    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)

    mask_veto = event.C1_C1VetoMask(event)
    mask_sele = event.C1_C1SelectionMask(event)
    mask_vetonosele = event.C1_C1VetoNoSelectionMask(event)
    out_mask_veto = awk.JaggedArray.fromiter(outputs["mask_veto"])
    out_mask_sele = awk.JaggedArray.fromiter(outputs["mask_sele"])
    out_mask_vetonosele = awk.JaggedArray.fromiter(outputs["mask_vetonosele"])

    assert np.array_equal(mask_veto.starts, out_mask_veto.starts)
    assert np.array_equal(mask_veto.stops, out_mask_veto.stops)
    assert np.array_equal(mask_veto.content, out_mask_veto.content)

    assert np.array_equal(mask_sele.starts, out_mask_sele.starts)
    assert np.array_equal(mask_sele.stops, out_mask_sele.stops)
    assert np.array_equal(mask_sele.content, out_mask_sele.content)

    assert np.array_equal(mask_vetonosele.starts, out_mask_vetonosele.starts)
    assert np.array_equal(mask_vetonosele.stops, out_mask_vetonosele.stops)
    assert np.array_equal(mask_vetonosele.content, out_mask_vetonosele.content)

def test_skimcollections_end(module):
    assert module.end() is None
    assert module.lambda_functions is None

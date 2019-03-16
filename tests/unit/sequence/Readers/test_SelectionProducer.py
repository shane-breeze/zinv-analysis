import pytest
import mock
import numpy as np
import awkward as awk

from zinv.sequence.Readers import SelectionProducer

class DummyEvent(object):
    def __init__(self):
        self.config = mock.MagicMock()
        self.dummy_selection = lambda ev: np.array([True, False, True])

        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.attribute_variation_sources = ['']
        self.cache = {}

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return SelectionProducer(event_selection_path="dummy_path.yaml")

def test_selectionproducer_init(module):
    assert module.event_selection_path == "dummy_path.yaml"

def test_selectionproducer_open(module, event):
    event.config.dataset.isdata = True
    data = """selections:\n    dummy_selection: \"ev: ev.dummy_selection(ev)\"\n"""\
        + """grouped_selections:\n    dummy_cat_sele:\n        - \"dummy_selection\"\n"""\
        + """cutflows:\n    dummy_cutflow:\n        dummy_cat:\n            - \"dummy_cat_sele\""""
    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)
    assert mocked_open.call_args_list == [mock.call("dummy_path.yaml", "r")]

def test_selectionproducer_begin_selections(module, event):
    event.config.dataset.isdata = True
    data = """selections:\n    dummy_selection: \"ev: ev.dummy_selection(ev)\"\n"""\
        + """grouped_selections:\n    dummy_cat_sele:\n        - \"dummy_selection\"\n"""\
        + """cutflows:\n    dummy_cutflow:\n        Data:\n            - \"dummy_cat_sele\""""
    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)

    assert module.selections == {
        'dummy_cutflow': {
            'Data': [('dummy_selection', 'ev: ev.dummy_selection(ev)')],
        },
        'dummy_cutflow_remove_dummy_selection': {
            'Data': [],
        },
    }

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "dummy_selection": [True, False, True],
        }, {
            "selection": [True, False, True],
            "selection_remove": [True, True, True],
        }],
    )
)
def test_selectionproducer_begin_evattr(module, event, inputs, outputs):
    event.dummy_selection = lambda ev: np.array(
        inputs["dummy_selection"], dtype=np.bool8,
    )
    event.size = len(inputs["dummy_selection"])
    event.config.dataset.isdata = True
    data = """selections:\n    dummy_selection: \"ev: ev.dummy_selection(ev)\"\n"""\
        + """grouped_selections:\n    dummy_cat_sele:\n        - \"dummy_selection\"\n"""\
        + """cutflows:\n    dummy_cutflow:\n        Data:\n            - \"dummy_cat_sele\""""
    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)

    for cutflow, outsele in [
        ("dummy_cutflow", outputs["selection"]),
        ("dummy_cutflow_remove_dummy_selection", outputs["selection_remove"]),
    ]:
        selection = getattr(event, "Cutflow_"+cutflow)(event)
        assert np.array_equal(selection, np.array(outsele))

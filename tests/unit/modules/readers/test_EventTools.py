import pytest
import mock
import numpy as np
import awkward as awk

from zinv.modules.readers import EventTools
from zinv.modules.readers.EventTools import get_size

class DummyEvent(object):
    def __init__(self):
        self.run = np.array([100001, 100002])
        self._nonbranch_cache = {}

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return EventTools(maxsize=int(6*1024**3))

@pytest.fixture()
def module_begin(module, event):
    module.begin(event)
    return module

def test_eventtools_begin(event, module):
    module.begin(event)
    assert event.nsig == 0
    assert event.source == ''
    assert "cache" in event._nonbranch_cache

def test_eventtools_event(event, module_begin):
    event.cache = mock.MagicMock()
    module_begin.event(event)
    assert event.nsig == 0
    assert event.source == ''
    assert event.cache.clear.call_args_list == [mock.call()]

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "isjagged": False,
            "array": [1., 2., 3.],
            "dtype": "float64",
        }, {
            "size": 24,
        }], [{
            "isjagged": False,
            "array": [1., 2., 3.],
            "dtype": "float32",
        }, {
            "size": 12,
        }], [{
            "isjagged": False,
            "array": [1., 2., 3.],
            "dtype": "int64",
        }, {
            "size": 24,
        }], [{
            "isjagged": False,
            "array": [1., 2., 3.],
            "dtype": "int32",
        }, {
            "size": 12,
        }], [{
            "isjagged": False,
            "array": [1., 0., 1.],
            "dtype": "bool8",
        }, {
            "size": 3,
        }], [{
            "isjagged": True,
            "array": [[1.], [2., 3.]],
            "dtype": "float64",
        }, {
            "size": 56,
        }], [{
            "isjagged": True,
            "array": [[1.], [2., 3.]],
            "dtype": "float32",
        }, {
            "size": 44,
        }], [{
            "isjagged": True,
            "array": [[1.], [2., 3.]],
            "dtype": "int64",
        }, {
            "size": 56,
        }], [{
            "isjagged": True,
            "array": [[1.], [2., 3.]],
            "dtype": "int32",
        }, {
            "size": 44,
        }], [{
            "isjagged": True,
            "array": [[1.], [0., 1.]],
            "dtype": "bool8",
        }, {
            "size": 35,
        }],
    )
)
def test_get_size(inputs, outputs):
    if inputs["isjagged"]:
        test_array = awk.JaggedArray.fromiter(
            inputs["array"],
        ).astype(getattr(np, inputs["dtype"]))
    else:
        test_array = np.array(
            inputs["array"], dtype=getattr(np, inputs["dtype"]),
        )
    assert get_size(test_array) == outputs["size"]

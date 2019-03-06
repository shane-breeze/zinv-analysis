import pytest
import mock
from cachetools import LRUCache

from sequence.Readers import EventTools

class DummyEvent(object):
    def __init__(self):
        self._callable_cache = {}

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
    assert "cache" in event._callable_cache

def test_eventtools_event(event, module_begin):
    event.cache = mock.MagicMock()
    module_begin.event(event)
    assert event.nsig == 0
    assert event.source == ''
    assert event.cache.clear.call_args_list == [mock.call()]

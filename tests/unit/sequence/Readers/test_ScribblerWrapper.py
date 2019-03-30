import pytest
import mock

from zinv.sequence.Readers import ScribblerWrapper

class DummyScribbler(object):
    def __init__(self):
        self.data = False
        self.dummy_attribute = "dummy attribute"
        self.name = "Dummy"

    def begin(self, event):
        self.begin_called = True

    def event(self, event):
        self.event_called = True

    def end(self):
        self.end_called = True

class DummyNoScribbler(object):
    def __init__(self):
        self.data = False
        self.dummy_attribute = "dummy attribute"
        self.name = "Dummy"

class DummyEvent(object):
    def __init__(self):
        self.config = mock.MagicMock()

@pytest.fixture()
def module():
    return ScribblerWrapper(DummyScribbler())

@pytest.fixture()
def event():
    return DummyEvent()

def test_scribblerwrap_init(module):
    assert module.data == False
    assert module.mc == True

def test_scribblerwrap_repr(module):
    assert repr(module) == repr(module.scribbler)

def test_scribblerwrap_getattr(module):
    assert module.dummy_attribute == "dummy attribute"

@pytest.mark.parametrize("attr", (["scribbler", "data", "mc"]))
def test_scribblerwrap_getattr_error(module, attr):
    delattr(module, attr)
    with pytest.raises(AttributeError) as err:
        print(getattr(module, attr))

def test_scribblerwrap_begin(module, event):
    event.config.dataset.isdata = False
    assert module.begin(event) is None
    assert module.isdata == False
    assert module.scribbler.begin_called == True

def test_scribblerwrap_begin_mismatch(module, event):
    event.config.dataset.isdata = True
    assert module.begin(event) == True
    assert module.isdata == True
    assert not hasattr(module.scribbler, 'begin_called')

def test_scribblerwrap_begin_notdata_notmc(module, event):
    event.config.dataset.isdata = False
    module.mc = False
    assert module.begin(event) == True
    assert module.isdata == False
    assert not hasattr(module.scribbler, 'begin_called')

def test_scribblerwrap_begin_nobegin(module, event):
    module.scribbler = DummyNoScribbler()
    event.config.dataset.isdata = False
    assert module.begin(event) is None
    assert module.isdata == False

def test_scribblerwrap_event(module, event):
    module.isdata = True
    module.data = True
    assert module.event(event) is None
    assert module.scribbler.event_called == True

def test_scribblerwrap_event_mismatch(module, event):
    module.isdata = True
    module.data = False
    assert module.event(event) == True
    assert not hasattr(module.scribbler, 'event_called')

def test_scribblerwrap_event_notdata_notmc(module, event):
    module.isdata = False
    module.mc = False
    assert module.event(event) == True
    assert not hasattr(module.scribbler, 'event_called')

def test_scribblerwrap_event_noevent(module, event):
    module.scribbler = DummyNoScribbler()
    module.isdata = True
    module.data = True
    assert module.event(event) == True

def test_scribblerwrap_end(module):
    assert module.end() is None
    assert module.scribbler.end_called == True

def test_scribblerwrap_end_noend(module):
    module.scribbler = DummyNoScribbler()
    assert module.end() == True

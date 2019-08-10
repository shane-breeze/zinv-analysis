import pytest
import mock
import numpy as np

from zinv.modules.readers import CertifiedLumiChecker

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.cache = {}
        self.run = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3], dtype=np.int32)
        self.luminosityBlock = np.array([0, 1, 2, 4, 5, 6, 0, 8, 9, 10, 100], dtype=np.int32)

    def register_function(self, event, name, function):
        self.__dict__[name] = function

    def hasbranch(self, branch):
        return hasattr(self, branch)

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def obj():
    return CertifiedLumiChecker(lumi_json_path="test.json")

def test_open(obj, event):
    data = """{"1": [[2, 5]], "2": [[0, 9], [11, 20]]}""".encode('utf-8')

    mock_file = mock.MagicMock()
    mock_file.read.return_value = data

    cxtmgr = mock.MagicMock()
    cxtmgr.__enter__.return_value = mock_file
    with mock.patch('urllib.request.urlopen', return_value=cxtmgr):
        obj.begin(event)

    assert True

def test_is_certified(obj, event):
    data = """{"1": [[2, 5]], "2": [[0, 9], [11, 20]]}""".encode('utf-8')

    mock_file = mock.MagicMock()
    mock_file.read.return_value = data

    cxtmgr = mock.MagicMock()
    cxtmgr.__enter__.return_value = mock_file
    with mock.patch('urllib.request.urlopen', return_value=cxtmgr):
        obj.begin(event)

    assert np.array_equal(
        event.IsCertified(event),
        np.array([False, False, True, True, True, False, True, True, True, False, False]),
    )

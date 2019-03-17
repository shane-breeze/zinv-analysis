import pytest
import mock
import numpy as np

from sequence.Readers import CertifiedLumiChecker

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.cache = {}
        self.run = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3], dtype=np.int32)
        self.luminosityBlock = np.array([0, 1, 2, 4, 5, 6, 0, 8, 9, 10, 100], dtype=np.int32)

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def obj():
    return CertifiedLumiChecker(lumi_json_path="test.json")

def test_open(obj, event):
    data = """{"1": [[2, 5]], "2": [[0, 9], [11, 20]]}"""
    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        obj.begin(event)
    assert mocked_open.call_args_list == [mock.call("test.json", "r")]

def test_is_certified(obj, event):
    data = """{"1": [[2, 5]], "2": [[0, 9], [11, 20]]}"""
    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        obj.begin(event)

    assert np.array_equal(
        event.IsCertified(event),
        np.array([False, False, True, True, True, False, True, True, True, False, False]),
    )

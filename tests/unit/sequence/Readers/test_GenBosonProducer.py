import pytest
import mock

from sequence.Readers import GenBosonProducer

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.nsig = 0
        self.source = ''
        self.cache = {}

    def delete_branches(self, branches):
        self.deleted_branches = branches

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return GenBosonProducer()

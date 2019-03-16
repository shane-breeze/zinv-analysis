import pytest
import mock

from zinv.sequence.Readers import TriggerChecker

def DummyEvent(object):
    def __init__(self):
        pass

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return TriggerChecker()

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "test_string": "ABC123_Run2016D_v4",
            "dataset": "ABC123",
            "run_letter": "D",
            "version": "4",
        }, {
            "result": True,
        }], [{
            "test_string": "ABC123_Run2016D_v4E",
        }, {
            "result": False,
        }], [{
            "test_string": "ABC123_Run2016D5_v4",
        }, {
            "result": False,
        }], [{
            "test_string": "ABC123_Run2017D5_v4",
        }, {
            "result": False,
        }],
    )
)
def test_triggerchecker_regex(module, inputs, outputs):
    match = module.regex.search(inputs["test_string"])
    assert bool(match) == outputs["result"]

    if match:
        assert match.group("dataset") == inputs["dataset"]
        assert match.group("run_letter") == inputs["run_letter"]
        assert match.group("version") == inputs["version"]

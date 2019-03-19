import pytest
import mock
import numpy as np

from zinv.sequence.Readers import TriggerChecker

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.config = mock.MagicMock()
        self.config.dataset.isdata = True
        self.config.dataset.name = "MET_Run2016F_v1"
        self.cache = {}

    def hasbranch(self, branch):
        return hasattr(self, branch)

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return TriggerChecker(
        trigger_selection_path = "dummy_path.yaml",
    )

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

def test_triggerchecker_open(module, event):
    data = """Data:\n    MET:\n        F1:\n            - 'HLT_PFMETNoMu90_PFMHTNoMu90_IDTight'"""
    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)
    assert mocked_open.call_args_list == [mock.call("dummy_path.yaml", "r")]

@pytest.mark.parametrize(
    "inputs", (
        {
            "isdata": True,
            "name": "MET_Run2016F_v1",
            "path": "HLT_PFMETNoMu90_PFMHTNoMu90_IDTight",
            "dataset": "MET",
            "trigger": [True, True, False, True, False],
        }, {
            "isdata": False,
            "name": "ZJetsToNu_Pt-250To400",
            "path": "HLT_IsoMu24",
            "dataset": "SingleMuon",
            "trigger": [False, False, True, True, True],
        },
    )
)
def test_triggerchecker_begin(module, event, inputs):
    data = """Data:\n    MET:\n        F1:\n            - 'HLT_PFMETNoMu90_PFMHTNoMu90_IDTight'\n"""\
            + """MC:\n    SingleMuon:\n        -'HLT_IsoMu24'"""
    event.config.dataset.isdata = inputs["isdata"]
    event.config.dataset.name = inputs["name"]
    event.size = len(inputs["trigger"])
    setattr(event, inputs["path"], np.array(inputs["trigger"], dtype=np.int32))

    mocked_open = mock.mock_open(read_data=data)
    with mock.patch("__builtin__.open", mocked_open):
        module.begin(event)

    trigger = getattr(event, "Is{}Triggered".format(inputs["dataset"]))(event)
    if inputs["isdata"]:
        outtrigger = getattr(event, inputs["path"])
    else:
        outtrigger = np.array([True]*event.size, dtype=np.int32)
        assert np.array_equal(trigger, outtrigger)

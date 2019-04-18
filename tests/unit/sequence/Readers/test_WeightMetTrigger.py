import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.sequence.Readers import WeightMetTrigger

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.attribute_variation_sources = []
        self.cache = {}

    def register_function(self, event, name, function):
        self.__dict__[name] = function

@pytest.fixture()
def path():
    toppath = os.path.abspath(os.environ["TOPDIR"])
    datapath = os.path.join(toppath, "zinv/data")
    return datapath

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module(path):
    return WeightMetTrigger(
        correction_files = {
            0: path+"/mettrigger/met_trigger_correction_0mu.txt",
            1: path+"/mettrigger/met_trigger_correction_1mu.txt",
            2: path+"/mettrigger/met_trigger_correction_2mu.txt",
        },
    )

def test_weightmettrigger_init(module, path):
    assert module.correction_files == {
        0: path+"/mettrigger/met_trigger_correction_0mu.txt",
        1: path+"/mettrigger/met_trigger_correction_1mu.txt",
        2: path+"/mettrigger/met_trigger_correction_2mu.txt",
    }
    assert module.cats == [0, 1, 2]
    assert all([hasattr(module, attr) for attr in [
        "xcents", "corr", "statup", "statdown", "systup", "systdown",
    ]])

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "met":      [-np.inf, -50., 0., 50.,     100.,    200.,        200.,         300., 1000.,  2000., np.inf, np.nan],
            "mucounts": [0,       0,    0,  0,       0,       1,           2,            3,    np.nan, 0,     0,      0],
            "nsig":     0.,
            "source":   "",
        }, {
            "eff":      [0., 0., 0., 0.00085, 0.11995, 0.954119033, 0.9128533589, 1., 1., 1., 1., np.nan],
        }], [{
            "met":      [-np.inf, -50., 0., 50.,     100.,    200.,        200.,         300., 1000.,  2000., np.inf, np.nan],
            "mucounts": [0,       0,    0,  0,       0,       1,           2,            3,    np.nan, 0,     0,      0],
            "nsig":     1.,
            "source":   "metTrigStat",
        }, {
            "eff":      [0., 0., 0., 0.000850255, 0.120748, 0.960702425, 0.9245834611, 1., 1., 1., 1., np.nan],
        }], [{
            "met":      [-np.inf, -50., 0., 50.,     100.,    200.,        200.,         300., 1000.,  2000., np.inf, np.nan],
            "mucounts": [0,       0,    0,  0,       0,       1,           2,            3,    np.nan, 0,     0,      0],
            "nsig":     -1.,
            "source":   "metTrigStat",
        }, {
            "eff":      [0., 0., 0., 0.000849785, 0.119164, 0.9473925085, 0.9009406438, 1., 1., 1., 1., np.nan],
        }], [{
            "met":      [-np.inf, -50., 0., 50.,     100.,    200.,        200.,         300., 1000.,  2000., np.inf, np.nan],
            "mucounts": [0,       0,    0,  0,       0,       1,           2,            3,    np.nan, 0,     0,      0],
            "nsig":     1.,
            "source":   "metTrigSyst",
        }, {
            "eff":      [0., 0., 0., 0.00085004, 0.1201, 0.9554547996, 0.915318063, 1., 1., 1., 1., np.nan],
        }], [{
            "met":      [-np.inf, -50., 0., 50.,     100.,    200.,        200.,         300., 1000.,  2000., np.inf, np.nan],
            "mucounts": [0,       0,    0,  0,       0,       1,           2,            3,    np.nan, 0,     0,      0],
            "nsig":     -1.,
            "source":   "metTrigSyst",
        }, {
            "eff":      [0., 0., 0., 0.00084996, 0.1198, 0.95278326635380, 0.91038865487585, 1., 1., 1., 1., np.nan],
        }],
    )
)
def test_weightmettrigger_begin(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    module.begin(event)

    event.METnoX_pt = mock.Mock(
        side_effect=lambda ev, source, nsig: np.array(inputs["met"], dtype=np.float32),
    )

    def mupt(ev, source, nsig, attr):
        assert attr == "pt"
        musele = DummyColl()
        musele.counts = np.array(inputs["mucounts"], dtype=np.float32)
        return musele
    event.MuonSelection = mock.Mock(side_effect=mupt)

    eff = event.WeightMETTrig(event, event.source, event.nsig)
    oeff = np.array(outputs["eff"], dtype=np.float32)
    print(eff)
    print(oeff)
    assert np.allclose(eff, oeff, rtol=1e-5, equal_nan=True)

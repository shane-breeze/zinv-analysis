import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.modules.readers import WeightObjects

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.attribute_variation_sources = []
        self.cache = {}

        self.Electron = DummyColl()

    def register_function(self, event, name, function):
        self.__dict__[name] = function

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return WeightObjects(
        correctors = [
            {
                "name": "eleIdIsoTight",
                "collection": "Electron",
                "binning_variables": ("ev, source, nsig: ev.Electron.eta", "ev, source, nsig: ev.Electron_ptShift(ev, source, nsig)"),
                "weighted_paths": [(1, "http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/electrons/electron_idiso_tight.csv")],
                "selection": ["CutBasedTightWP", "eta_pt"],
                "add_syst": "ev, source, nsig: awk.JaggedArray.zeros_like(ev.Electron.eta)",
                "nuisances": ["eleIdIsoTight", "eleEnergyScale"],
            }, {
                "name": "eleIdIsoVeto",
                "collection": "Electron",
                "binning_variables": ("ev, source, nsig: ev.Electron.eta", "ev, source, nsig: ev.Electron_ptShift(ev, source, nsig)"),
                "weighted_paths": [(1, "http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/electrons/electron_idiso_veto.csv")],
                "selection": ["CutBasedVetoWP", "eta_pt"],
                "add_syst": "ev, source, nsig: awk.JaggedArray.zeros_like(ev.Electron.eta)",
                "nuisances": ["eleIdIsoVeto", "eleEnergyScale"],
            }, {
                "name": "eleReco",
                "collection": "Electron",
                "binning_variables": ("ev, source, nsig: ev.Electron.eta", "ev, source, nsig: ev.Electron_ptShift(ev, source, nsig)"),
                "weighted_paths": [(1, "http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/electrons/electron_reconstruction.csv")],
                "selection": ["Reco", "eta_pt"],
                "add_syst": "ev, source, nsig: 0.01*((ev.Electron_ptShift(ev, source, nsig)<20) | (ev.Electron_ptShift(ev, source, nsig)>80))",
                "nuisances": ["eleReco", "eleEnergyScale"],
            }, {
                "name": "eleTrig",
                "collection": "Electron",
                "binning_variables": ("ev, source, nsig: ev.Electron_ptShift(ev, source, nsig)", "ev, source, nsig: np.abs(ev.Electron.eta)"),
                "weighted_paths": [(1, "http://www.hep.ph.ic.ac.uk/~sdb15/Analysis/ZinvWidth/data/electrons/electron_trigger_v2.csv")],
                "selection": ["HLT_Ele27_WPTight_Gsf", "pt_eta"],
                "add_syst": "ev, source, nsig: awk.JaggedArray.zeros_like(ev.Electron.eta)",
                "nuisances": ["eleTrig", "eleEnergyScale"],
            },
        ],
    )

def test_weightobjects_init(module):
    assert hasattr(module, "correctors")

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "name":   "eleIdIsoTight",
            "eleeta": [[0.1], [-3., -0.5, 0.5, 3], [-1e6, 1e6], [0.1, 0.3]],
            "elept":  [[10.], [20., 30., 40., 50], [100., 120.], [-1e6, 1e6]],
            "nsig":   0.,
            "source": "",
        }, {
            "sf": [[9.4587630033e-01], [8.8245934248e-01, 9.5299839973e-01, 9.7981154919e-01, 9.3766236305e-01], [1.0510855913e+00, 1.0213568211e+00], [9.4587630033e-01, 1.0118906498e+00]],
        }], [{
            "name":   "eleIdIsoVeto",
            "eleeta": [[0.1], [-3., -0.5, 0.5, 3], [-1e6, 1e6], [0.1, 0.3]],
            "elept":  [[10.], [20., 30., 40., 50], [100., 120.], [-1e6, 1e6]],
            "nsig":   1.,
            "source": "eleIdIsoVeto",
        }, {
            "sf": [[9.96433724056E-01], [9.85382607704E-01, 9.94656152779E-01, 9.91259808420E-01, 1.00426236358E+00], [1.04882350978E+00, 1.04453718557E+00], [9.96433724056E-01, 1.01499641427E+00]],
        }], [{
            "name":   "eleReco",
            "eleeta": [[0.1], [-3., -0.5, 0.5, 3], [-1e6, 1e6], [0.1, 0.3]],
            "elept":  [[10.], [20., 30., 40., 50], [100., 120.], [-1e6, 1e6]],
            "nsig":   -1.,
            "source": "eleReco",
        }, {
            "sf": [[9.73408456347E-01], [1.29357287767E+00, 9.78643468014E-01, 9.81689022011E-01, 8.90130249373E-01], [1.29019777985E+00, 8.87807785202E-01], [9.73408456347E-01, 9.77046870735E-01]],
        }],
    )
)
def test_weightobjects_begin(module, event, inputs, outputs):
    event.source = inputs["source"]
    event.nsig = inputs["nsig"]

    module.begin(event)

    eleeta = awk.JaggedArray.fromiter(inputs["eleeta"]).astype(np.float32)
    event.Electron.eta = eleeta
    event.Electron_eta = eleeta

    elept = awk.JaggedArray.fromiter(inputs["elept"]).astype(np.float32)
    event.Electron.ptShift = mock.Mock(side_effect=lambda ev, source, nsig: elept)
    event.Electron_ptShift = mock.Mock(side_effect=lambda ev, source, nsig: elept)

    sf = getattr(event, "Electron_Weight{}SF".format(inputs["name"]))(event, event.source, event.nsig)
    osf = awk.JaggedArray.fromiter(outputs["sf"]).astype(np.float32)

    print(sf)
    print(osf)
    assert np.array_equal(sf.starts, osf.starts)
    assert np.array_equal(sf.stops, osf.stops)
    assert np.allclose(
        sf.content, osf.content, rtol=1e-6, equal_nan=True,
    )

def test_weightobjects_end(module):
    assert module.end() is None
    assert module.lambda_functions is None

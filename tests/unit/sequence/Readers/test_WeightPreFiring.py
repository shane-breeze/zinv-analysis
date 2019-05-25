import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.sequence.Readers import WeightPreFiring

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.attribute_variation_sources = []
        self.cache = {}

        self.Jet = DummyColl()
        self.Photon = DummyColl()

    def register_function(self, event, name, function):
        self.__dict__[name] = function

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def path():
    toppath = os.path.abspath(os.environ["TOPDIR"])
    datapath = os.path.join(toppath, "zinv/data")
    return datapath

@pytest.fixture()
def module(path):
    return WeightPreFiring(
        jet_eff_map_path = path+"/prefiring/L1prefiring_jetpt_2016BtoH.txt",
        photon_eff_map_path = path+"/prefiring/L1prefiring_photonpt_2016BtoH.txt",
        jet_selection = "ev, source, nsig: (ev.Jet_ptShift(ev, source, nsig)>20) & ((2<np.abs(ev.Jet_eta)) & (np.abs(ev.Jet_eta)<3))",
        photon_selection = "ev, source, nsig: (ev.Photon_ptShift(ev, source, nsig)>20) & ((2<np.abs(ev.Photon_eta)) & (np.abs(ev.Photon_eta)<3))",
        syst = 0.2,
    )

def test_weightprefiring_init(module, path):
    assert module.jet_eff_map_path == path + "/prefiring/L1prefiring_jetpt_2016BtoH.txt"
    assert module.photon_eff_map_path == path + "/prefiring/L1prefiring_photonpt_2016BtoH.txt"
    assert module.jet_selection == "ev, source, nsig: (ev.Jet_ptShift(ev, source, nsig)>20) & ((2<np.abs(ev.Jet_eta)) & (np.abs(ev.Jet_eta)<3))"
    assert module.photon_selection == "ev, source, nsig: (ev.Photon_ptShift(ev, source, nsig)>20) & ((2<np.abs(ev.Photon_eta)) & (np.abs(ev.Photon_eta)<3))"
    assert module.syst == 0.2

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "jpt":  [[], [5., 10., 32.5], [50., 100., 400., 1000.]],
            "jeta": [[], [-2.1, 2.2, 2.3], [2.4, 0., 2.5, 2.6]],
            "jphi": [[], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]],
            "ppt":  [[], [5., 10., 15.], [50., 100., 400., 1000.]],
            "peta": [[], [-2.1, 2.2, 2.3], [2.4, 0., 2.5, 2.6]],
            "pphi": [[], [1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 0.7]],
            "source": "",
            "nsig": 0.,
        }, {
            "wpref": [1., 0.99165673500000000, 0.04580499402533940],
        }], [{
            "jpt":  [[], [5., 10., 32.5], [50., 100., 400., 1000.]],
            "jeta": [[], [-2.1, 2.2, 2.3], [2.4, 0., 2.5, 2.6]],
            "jphi": [[], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]],
            "ppt":  [[], [5., 10., 15.], [50., 100., 400., 1000.]],
            "peta": [[], [-2.1, 2.2, 2.3], [2.4, 0., 2.5, 2.6]],
            "pphi": [[], [1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 0.7]],
            "source": "prefiring",
            "nsig": 1.,
        }, {
            "wpref": [1., 0.98905741742909600, 0.01120148311952490],
        }], [{
            "jpt":  [[], [5., 10., 32.5], [50., 100., 400., 1000.]],
            "jeta": [[], [-2.1, 2.2, 2.3], [2.4, 0., 2.5, 2.6]],
            "jphi": [[], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]],
            "ppt":  [[], [5., 10., 15.], [50., 100., 400., 1000.]],
            "peta": [[], [-2.1, 2.2, 2.3], [2.4, 0., 2.5, 2.6]],
            "pphi": [[], [1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 0.7]],
            "source": "prefiring",
            "nsig": -1.,
        }, {
            "wpref": [1., 0.99425605257090400, 0.11686173455535100],
        }], [{
            "jpt":  [[], [5., 10., 32.5], [50., 100., 400., 1000.]],
            "jeta": [[], [-2.1, 2.2, 2.3], [2.4, 0., 2.5, 2.6]],
            "jphi": [[], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]],
            "ppt":  [[], [5., 10., 15.], [50., 100., 400., 1000.]],
            "peta": [[], [-2.1, 2.2, 2.3], [2.4, 0., 2.5, 2.6]],
            "pphi": [[], [1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 0.7]],
            "source": "other",
            "nsig": -1.,
        }, {
            "wpref": [1., 0.99165673500000000, 0.04580499402533940],
        }],
    )
)
def test_weightprefiring_begin(module, event, inputs, outputs):
    event.nsig = inputs["nsig"]
    event.source = inputs["source"]

    jpt = awk.JaggedArray.fromiter(inputs["jpt"]).astype(np.float32)
    jeta = awk.JaggedArray.fromiter(inputs["jeta"]).astype(np.float32)
    jphi = awk.JaggedArray.fromiter(inputs["jphi"]).astype(np.float32)

    ppt = awk.JaggedArray.fromiter(inputs["ppt"]).astype(np.float32)
    peta = awk.JaggedArray.fromiter(inputs["peta"]).astype(np.float32)
    pphi = awk.JaggedArray.fromiter(inputs["pphi"]).astype(np.float32)

    event.Jet.ptShift = mock.Mock(side_effect=lambda ev, source, nsig: jpt)
    event.Jet.eta = jeta
    event.Jet.phi = jphi
    event.Jet_ptShift = mock.Mock(side_effect=lambda ev, source, nsig: jpt)
    event.Jet_eta = jeta
    event.Jet_phi = jphi

    event.Photon.ptShift = mock.Mock(side_effect=lambda ev, source, nsig: ppt)
    event.Photon.eta = peta
    event.Photon.phi = pphi
    event.Photon_ptShift = mock.Mock(side_effect=lambda ev, source, nsig: ppt)
    event.Photon_eta = peta
    event.Photon_phi = pphi

    module.begin(event)

    wpref = event.WeightPreFiring(event, event.source, event.nsig)
    owpref = np.array(outputs["wpref"], dtype=np.float32)

    print(wpref)
    print(owpref)
    assert np.allclose(wpref, owpref, rtol=1e-6, equal_nan=True)

def test_weightprefiring_end(module):
    assert module.end() is None
    assert module.functions is None

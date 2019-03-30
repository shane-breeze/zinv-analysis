import pytest
import mock
import os
import numpy as np
import awkward as awk

from zinv.sequence.Readers import GenBosonProducer

class DummyColl(object):
    pass

class DummyEvent(object):
    def __init__(self):
        self.iblock = 0
        self.source = ''
        self.nsig = 0.
        self.cache = {}
        self.config = mock.MagicMock()

        self.GenPart = DummyColl()
        self.GenDressedLepton = DummyColl()

    def delete_branches(self, branches):
        pass

@pytest.fixture()
def event():
    return DummyEvent()

@pytest.fixture()
def module():
    return GenBosonProducer()

@pytest.mark.parametrize(
    "inputs,outputs", (
        [{
            "gp_pdg":       [[11,   -11],  [12,   -11],  [23],   [24]],
            "gp_pt":        [[100., 110.], [120., 130.], [140.], [150.]],
            "gp_eta":       [[0.1,  0.3],  [0.5,  0.7],  [0.9],  [1.1]],
            "gp_phi":       [[-2.1, -1.5], [-0.5, 0.5],  [1.6],  [2.4]],
            "gp_mass":      [[0.,   0.],   [0.,   0.],   [80.],  [91]],
            "gp_flags":     [[257,  257],  [257,  257],  [257],  [257]],
            "gp_status":    [[1,    1],    [1,    1],    [23],   [23]],
            "gp_motheridx": [[0,    0],    [0,    0],    [0],    [0]],
            "dl_pdg":       [[11,   -11],  [11,   -11],  [],     []],
            "dl_pt":        [[105., 114.], [123., 132.], [],     []],
            "dl_eta":       [[0.1,  0.3],  [1.9,  0.7],  [],     []],
            "dl_phi":       [[-2.1, -1.5], [-0.5, 0.5],  [],     []],
            "dl_mass":      [[0.,   0.],   [0.,   0.],   [],     []],
        }, {
            "ngenbosons": [1, 1, 1, 1],
            "vpt":   [209.2355959700400, 221.2256247587580, 0., 0.],
            "veta":  [0.2145319040114,   0.6813357781853,   0., 0.],
            "vphi":  [-1.7872882361237,  0.0260085382506,   0., 0.],
            "vmass": [68.2778405383693,  123.2839720649430, 0., 0.],
        }],
    )
)
def test_genbosonproducer_event(module, event, inputs, outputs):
    event.GenPart.pdgId = awk.JaggedArray.fromiter(inputs["gp_pdg"]).astype(np.int32)
    event.GenPart.pt = awk.JaggedArray.fromiter(inputs["gp_pt"]).astype(np.float32)
    event.GenPart.eta = awk.JaggedArray.fromiter(inputs["gp_eta"]).astype(np.float32)
    event.GenPart.phi = awk.JaggedArray.fromiter(inputs["gp_phi"]).astype(np.float32)
    event.GenPart.mass = awk.JaggedArray.fromiter(inputs["gp_mass"]).astype(np.float32)
    event.GenPart.statusFlags = awk.JaggedArray.fromiter(inputs["gp_flags"]).astype(np.int32)
    event.GenPart.status = awk.JaggedArray.fromiter(inputs["gp_status"]).astype(np.int32)
    event.GenPart.genPartIdxMother = awk.JaggedArray.fromiter(inputs["gp_motheridx"]).astype(np.int32)

    event.GenDressedLepton.pdgId = awk.JaggedArray.fromiter(inputs["dl_pdg"]).astype(np.int32)
    event.GenDressedLepton.pt = awk.JaggedArray.fromiter(inputs["dl_pt"]).astype(np.float32)
    event.GenDressedLepton.eta = awk.JaggedArray.fromiter(inputs["dl_eta"]).astype(np.float32)
    event.GenDressedLepton.phi = awk.JaggedArray.fromiter(inputs["dl_phi"]).astype(np.float32)
    event.GenDressedLepton.mass = awk.JaggedArray.fromiter(inputs["dl_mass"]).astype(np.float32)

    module.event(event)

    assert np.array_equal(
        event.nGenBosons, np.array(outputs["ngenbosons"], dtype=np.int32),
    )
    assert np.allclose(
        event.GenPartBoson.pt, np.array(outputs["vpt"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )
    assert np.allclose(
        event.GenPartBoson.eta, np.array(outputs["veta"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )
    assert np.allclose(
        event.GenPartBoson.phi, np.array(outputs["vphi"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )
    assert np.allclose(
        event.GenPartBoson.mass, np.array(outputs["vmass"], dtype=np.float32),
        rtol=1e-6, equal_nan=True,
    )

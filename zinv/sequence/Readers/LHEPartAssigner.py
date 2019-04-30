import numpy as np
from zinv.utils.AwkwardOps import get_nth_object

class LHEPartAssigner(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        event.register_function(event, "LeptonIs", leptonis)

def leptonis(ev, attr):
    if not ev.hasbranch("LeptonIs{}".format(attr)):
        if not ev.hasbranch("LHEPart_pdgId"):
            return np.zeros(ev.size, dtype=np.bool8)
        pdg = ev.LHEPart.pdgId
        lepton_decay = get_nth_object(np.abs(
            pdg[(np.abs(pdg)==11) | (np.abs(pdg)==13) | (np.abs(pdg)==15)]
        ), 0, ev.size)
        ev.LeptonIsElectron = (lepton_decay == 11)
        ev.LeptonIsMuon = (lepton_decay == 13)
        ev.LeptonIsTau = (lepton_decay == 15)
    return getattr(ev, "LeptonIs{}".format(attr))

class GenPartAssigner(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def begin(self, event):
        event.register_function(event, "nGenTauL", ngen_taul)

def ngen_taul(ev):
    if not ev.hasbranch("nGenTauL_val"):
        flag = ev.GenPart.statusFlags
        pdgs = ev.GenPart.pdgId

        ev.nGenTauL_val = (
            (flag&(1<<10)==(1<<10)) & ((np.abs(pdgs)==11) | (np.abs(pdgs)==13))
        ).sum()
    return ev.nGenTauL_val

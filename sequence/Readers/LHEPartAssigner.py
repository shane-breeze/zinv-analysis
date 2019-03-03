import numpy as np

class LHEPartAssigner(object):
    old_parents = ["WJetsToLNu", "DYJetsToLL", "ZJetsToLL", "GStarJetsToLL"]
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def event(self, event):
        if event.config.dataset.parent not in self.old_parents:
            return True

        pdg = event.LHEPart.pdgId
        event.LeptonDecay = np.abs(
            pdg[(np.abs(pdg)==11) | (np.abs(pdg)==13) | (np.abs(pdg)==15)]
        )[:,0]
        event.LeptonIsElectron = (event.LeptonDecay == 11)
        event.LeptonIsMuon = (event.LeptonDecay == 13)
        event.LeptonIsTau = (event.LeptonDecay == 15)
        event.delete_branches(["LHEPart_pdgId"])

class GenPartAssigner(object):
    old_parents = ["WJetsToLNu", "DYJetsToLL", "ZJetsToLL", "GStarJetsToLL"]
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    def event(self, event):
        if event.config.dataset.parent not in self.old_parents:
            return True

        flag = event.GenPart.statusFlags
        pdgs = event.GenPart.pdgId

        event.nGenTauL = (
            (flag&(1<<10)==(1<<10)) & ((np.abs(pdgs)==11) | (np.abs(pdgs)==13))
        ).sum()

        event.delete_branches(["GenPart_pdgId", "GenPart_statusFlags"])

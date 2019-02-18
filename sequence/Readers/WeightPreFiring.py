import pandas as pd
import numpy as np
import awkward
from numba import njit, float32

from utils.NumbaFuncs import get_bin_indices
from utils.Geometry import DeltaR2
from utils.Lambda import Lambda

from . import Collection

class WeightPreFiring(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.jet_eff_map = get_maps(self.jet_eff_map_path)
        self.photon_eff_map = get_maps(self.photon_eff_map_path)

    def begin(self, event):
        self.functions = {
            self.jet_selection: Lambda(self.jet_selection),
            self.photon_selection: Lambda(self.photon_selection),
        }

    def end(self):
        self.functions = {}

    def event(self, event):
        if event.config.dataset.isdata:
            event.Weight_PreFiring = np.ones(event.size)
            event.Weight_PreFiringUp = np.ones(event.size)
            event.Weight_PreFiringDown = np.ones(event.size)
            return True

        jets = Collection("JetsForPreFiring", event, "Jet", event.Jet(self.functions[self.jet_selection]))
        jet_effs, jet_effs_stat = get_efficiencies(jets, self.jet_eff_map)
        jet_effs_err = awkward.JaggedArray(
            jets.starts, jets.stops,
            np.sqrt(jet_effs_stat.content**2 + (self.syst*jet_effs.content)**2),
        )
        jets.preFiringEff = jet_effs
        jets.preFiringEffErr = jet_effs_err

        photons = Collection("PhotonsForPreFiring", event, "Photon", event.Photon(self.functions[self.photon_selection]))
        photon_effs, photon_effs_stat = get_efficiencies(photons, self.photon_eff_map)
        photon_effs_err = awkward.JaggedArray(
            photons.starts, photons.stops,
            np.sqrt(photon_effs_stat.content**2 + (self.syst*photon_effs.content)**2),
        )
        photons.preFiringEff = photon_effs
        photons.preFiringEffErr = photon_effs_err

        p_nonprefiring, p_nonprefiring_up, p_nonprefiring_down = get_event_nonprefiring_prob(photons, jets)
        event.Weight_PreFiring = p_nonprefiring
        event.Weight_PreFiringUp = p_nonprefiring_up / p_nonprefiring
        event.Weight_PreFiringDown = p_nonprefiring_down / p_nonprefiring

        if self.apply:
            event.Weight_MET *= p_nonprefiring
            event.Weight_SingleMuon *= p_nonprefiring
            event.Weight_SingleElectron *= p_nonprefiring

def get_maps(path):
    df = pd.read_table(path, sep='\s+')
    df.loc[df["yupp"]==df["yupp"].max(), "yupp"] = np.inf # overflow pt (y-axis)
    return df

def get_efficiencies(objects, effmap):
    xlow = np.unique(effmap["xlow"].values)
    xupp = np.unique(effmap["xupp"].values)
    xind = get_bin_indices(objects.eta.content, xlow, xupp)

    ylow = np.unique(effmap["ylow"].values)
    yupp = np.unique(effmap["yupp"].values)
    yind = get_bin_indices(objects.pt.content, ylow, yupp)

    indices = ylow.shape[0]*xind + yind
    df = effmap.iloc[indices]
    return awkward.JaggedArray(
        objects.starts, objects.stops, df["content"].values,
    ), awkward.JaggedArray(
        objects.starts, objects.stops,
        df["error"].values,
    )

def get_event_nonprefiring_prob(photons, jets):
    return njit_get_event_nonprefiring_prob(
        photons.eta.content, photons.phi.content,
        photons.preFiringEff.content, photons.preFiringEffErr.content,
        photons.starts, photons.stops,
        jets.eta.content, jets.phi.content,
        jets.preFiringEff.content, jets.preFiringEffErr.content,
        jets.starts, jets.stops,
    )

@njit
def njit_get_event_nonprefiring_prob(
    pho_eta, pho_phi, pho_p, pho_perr, pho_stas, pho_stos,
    jet_eta, jet_phi, jet_p, jet_perr, jet_stas, jet_stos,
):
    nonprefiring_prob = np.ones(pho_stas.shape[0], dtype=float32)
    nonprefiring_prob_up = np.ones(pho_stas.shape[0], dtype=float32)
    nonprefiring_prob_down = np.ones(pho_stas.shape[0], dtype=float32)

    for iev, (psta, psto, jsta, jsto) in enumerate(zip(pho_stas, pho_stos,
                                                       jet_stas, jet_stos)):
        j_skip = []
        for pidx in range(psta, psto):
            pprob = max(0., min(1., pho_p[pidx]))
            pprob_up = max(0., min(1., pho_p[pidx] + pho_perr[pidx]))
            pprob_down = max(0., min(1., pho_p[pidx] - pho_perr[pidx]))

            match_jidx = jsta-1
            min_dr2 = 0.16
            for jidx in range(jsta, jsto):
                deta = pho_eta[pidx] - jet_eta[jidx]
                dphi = pho_phi[pidx] - jet_phi[jidx]

                dr2 = DeltaR2(deta, dphi)
                if dr2 < min_dr2:
                    min_dr2 = dr2
                    match_jidx = jidx

            if jsta <= match_jidx and match_jidx < jsto:
                j_skip.append(jidx)
                jprob = max(0., min(1., jet_p[jidx]))
                jprob_up = max(0., min(1., jet_p[jidx] + jet_perr[jidx]))
                jprob_down = max(0., min(1., jet_p[jidx] - jet_perr[jidx]))

                pprob = max(pprob, jprob)
                pprob_up = max(pprob_up, jprob_up)
                pprob_down = max(pprob_down, jprob_down)

            nonprefiring_prob[iev] *= (1-pprob)
            nonprefiring_prob_up[iev] *= (1-pprob_up)
            nonprefiring_prob_down[iev] *= (1-pprob_down)

        for jidx in range(jsta, jsto):
            if jidx in j_skip:
                continue

            jprob = max(0., min(1., jet_p[jidx]))
            jprob_up = max(0., min(1., jet_p[jidx] + jet_perr[jidx]))
            jprob_down = max(0., min(1., jet_p[jidx] - jet_perr[jidx]))

            nonprefiring_prob[iev] *= (1-jprob)
            nonprefiring_prob_up[iev] *= (1-jprob_up)
            nonprefiring_prob_down[iev] *= (1-jprob_down)

    return nonprefiring_prob, nonprefiring_prob_up, nonprefiring_prob_down

if __name__ == "__main__":
    weight_prefiring = WeightPreFiring(
        jet_eff_map_path = "/vols/build/cms/sdb15/zinv-analysis/data/prefiring/L1prefiring_jetpt_2016BtoH.txt",
        photon_eff_map_path = "/vols/build/cms/sdb15/zinv-analysis/data/prefiring/L1prefiring_photonpt_2016BtoH.txt",
        jet_selection = "j: (j.pt>20) & ((2<np.abs(j.eta)) & (np.abs(j.eta)<3))",
        photon_selection = "y: (y.pt>20) & ((2<np.abs(y.eta)) & (np.abs(y.eta)<3))",
        syst = 0.2,
        apply = False,
    )

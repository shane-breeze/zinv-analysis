import pandas as pd
import numpy as np
import numba as nb
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.NumbaFuncs import weight_numba, get_bin_indices
from zinv.utils.Geometry import DeltaR2
from zinv.utils.Lambda import Lambda

def evaluate_prefiring_weight(ev, source, nsig, funcs, jetmap, photmap, syst):
    @nb.njit(["UniTuple(float32[:],3)(float32[:],float32[:],float32[:],float32[:],int64[:],int64[:],float32[:],float32[:],float32[:],float32[:],int64[:],int64[:])"])
    def prob_nonprefiring_numba(
        pho_eta, pho_phi, pho_p, pho_perr, pho_stas, pho_stos,
        jet_eta, jet_phi, jet_p, jet_perr, jet_stas, jet_stos,
    ):
        nonprefiring_prob = np.ones(pho_stas.shape[0], dtype=np.float32)
        nonprefiring_prob_up = np.ones(pho_stas.shape[0], dtype=np.float32)
        nonprefiring_prob_down = np.ones(pho_stas.shape[0], dtype=np.float32)

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

    jet_mask = funcs["Jet"](ev, source, nsig)
    jet_eta = ev.Jet_eta[jet_mask]
    jet_effs, jet_effs_stat = get_efficiencies(
        ev, source, nsig, "Jet", jet_mask, jetmap,
    )
    jet_effs_err = np.sqrt(jet_effs_stat**2 + (syst*jet_effs)**2)

    photon_mask = funcs["Photon"](ev, source, nsig)
    photon_eta = ev.Photon_eta[photon_mask]
    photon_effs, photon_effs_stat = get_efficiencies(
        ev, source, nsig, "Photon", photon_mask, photmap,
    )
    photon_effs_err = np.sqrt(photon_effs_stat**2 + (syst*photon_effs)**2)

    prob, probup, probdown = prob_nonprefiring_numba(
        photon_eta.content, ev.Photon_phi[photon_mask].content,
        photon_effs.astype(np.float32), photon_effs_err.astype(np.float32),
        photon_eta.starts, photon_eta.stops,
        jet_eta.content, ev.Jet_phi[jet_mask].content,
        jet_effs.astype(np.float32), jet_effs_err.astype(np.float32),
        jet_eta.starts, jet_eta.stops,
    )
    return weight_numba(
        prob, nsig,
        (source=="prefiring")*(probup/prob-1.),
        (source=="prefiring")*(probdown/prob-1.),
    )

class WeightPreFiring(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.jet_eff_map = get_maps(self.jet_eff_map_path)
        self.photon_eff_map = get_maps(self.photon_eff_map_path)

    def begin(self, event):
        functions = {
            "Jet": Lambda(self.jet_selection),
            "Photon": Lambda(self.photon_selection),
        }
        event.register_function(
            event, "WeightPreFiring", partial(
                evaluate_prefiring_weight, funcs=functions,
                jetmap=self.jet_eff_map, photmap=self.photon_eff_map,
                syst=self.syst,
            ),
        )

    def end(self):
        self.functions = None

def get_maps(path):
    df = pd.read_csv(path, sep='\s+')
    df.loc[df["yupp"]==df["yupp"].max(), "yupp"] = np.inf # overflow pt (y-axis)
    return df

def get_efficiencies(event, source, nsig, obj, selection, effmap):
    indices = get_bin_indices(
        [getattr(event, "{}_eta".format(obj))[selection].content,
         getattr(event, "{}_ptShift".format(obj))(event, source, nsig)[selection].content.astype(np.float32)],
        [effmap["xlow"].values, effmap["ylow"].values],
        [effmap["xupp"].values, effmap["yupp"].values],
        1,
    )[:,0]
    df = effmap.iloc[indices]
    return df["content"].values, df["error"].values

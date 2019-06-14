import re
import numpy as np
import numba as nb
import pandas as pd
import awkward as awk
from functools import partial

from zinv.utils.NumbaFuncs import get_bin_indices, event_to_object_var, interpolate
from zinv.utils.Geometry import RadToCart2D, CartToRad2D, DeltaR2

@nb.njit(["float32[:](float32[:],float32[:],float32[:],float32[:],float32[:])",
          "float64[:](float64[:],float64[:],float64[:],float64[:],float64[:])"])
def jer_formula(x, p0, p1, p2, p3):
    return np.sqrt(p0*np.abs(p0)/(x*x)+p1*p1*np.power(x,p3)+p2*p2)

def met_shift(ev):
    @nb.njit(["UniTuple(float32[:],2)(float32[:],float32[:],float32[:],float32[:],float32[:],int64[:],int64[:])"])
    def met_shift_numba(met, mephi, jpt, jptshift, jphi, jstarts, jstops):
        jpx_old, jpy_old = RadToCart2D(jpt, jphi)
        jpx_new, jpy_new = RadToCart2D(jptshift, jphi)

        mex, mey = RadToCart2D(met[:], mephi[:])
        for iev, (start, stop) in enumerate(zip(jstarts, jstops)):
            for iob in range(start, stop):
                mex[iev] += (jpx_old[iob] - jpx_new[iob])
                mey[iev] += (jpy_old[iob] - jpy_new[iob])

        return CartToRad2D(mex, mey)
    return met_shift_numba(
        ev.MET_ptJESOnly, ev.MET_phiJESOnly, ev.Jet_ptJESOnly.content,
        ev.Jet_pt.content, ev.Jet_phi.content,
        ev.Jet_pt.starts, ev.Jet_pt.stops,
    )

def met_sumet_shift(ev):
    @nb.jit(["float32[:](float32[:], float32[:], float32[:], int64[:], int64[:])"])
    def nb_met_sumet_shift(sumet, jpt, cjpt, jstas, jstos):
        csumet = np.zeros_like(sumet, dtype=np.float32)
        for iev, (start, stop) in enumerate(zip(jstas, jstos)):
            csumet[iev] = sumet[iev] + (cjpt[start:stop] - jpt[start:stop]).sum()
        return csumet

    return nb_met_sumet_shift(
        ev.MET_sumEtJESOnly,
        ev.Jet_ptJESOnly.content, ev.Jet_pt.content,
        ev.Jet_pt.starts, ev.Jet_pt.stops,
    )

def match_jets_from_genjets(event, maxdr, ndpt):
    @nb.njit(["int64[:](float32[:],float32[:],float32[:],float32[:],int64[:],int64[:],float32[:],float32[:],float32[:],int64[:],int64[:],float32,float32)"])
    def numba_function(
        jpt, jeta, jphi, jres, jsta, jsto,
        gjpt, gjeta, gjphi, gjsta, gjsto,
        maxdr_, ndpt_,
    ):
        match_idx = -1*np.ones_like(jpt, dtype=np.int64)
        for iev, (jb, je, gjb, gje) in enumerate(zip(jsta, jsto, gjsta, gjsto)):
            for ijs in range(jb, je):
                for igjs in range(gjb, gje):
                    within_dr2 = DeltaR2(
                        jeta[ijs]-gjeta[igjs],
                        jphi[ijs]-gjphi[igjs],
                    ) < maxdr_**2
                    within_dpt = np.abs(jpt[ijs]-gjpt[igjs]) < ndpt_*jres[ijs]*jpt[ijs]
                    if within_dr2 and within_dpt:
                        match_idx[ijs] = igjs-gjb
                        break

        return match_idx

    return awk.JaggedArray(
        event.Jet.pt.starts,
        event.Jet.pt.stops,
        numba_function(
            event.Jet.pt.content, event.Jet.eta.content, event.Jet.phi.content,
            event.Jet_ptResolution(event).content,
            event.Jet.pt.starts, event.Jet.pt.stops,
            event.GenJet.pt.content, event.GenJet.eta.content,
            event.GenJet.phi.content,
            event.GenJet.pt.starts, event.GenJet.pt.stops,
            maxdr, ndpt,
        ),
    )

class JecVariations(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.jesuncs = read_table(
            self.jes_unc_file,
            underflow_cols=["eta_low"], overflow_cols=["eta_high"],
            csv=True,
        )

        self.jersfs = read_table(
            self.jer_sf_file,
            underflow_cols=["eta_low"], overflow_cols=["eta_high"],
        )
        self.jers = read_table(
            self.jer_file,
            underflow_cols=["eta_low", "rho_low"],
            overflow_cols=["eta_high", "rho_high"],
        )

    def begin(self, event):
        np.random.seed(123456+event.config.dataset.idx)

        self.jesuncs["pt"] = self.jesuncs["pt"].apply(lambda x: list(eval(x.replace("nan", "np.nan"))))
        self.jesuncs["corr_up"] = self.jesuncs["corr_up"].apply(lambda x: list(eval(x.replace("nan", "np.nan"))))
        self.jesuncs["corr_down"] = self.jesuncs["corr_down"].apply(lambda x: list(eval(x.replace("nan", "np.nan"))))
        self.jes_sources = self.jesuncs["source"].unique()
        event.JesSources = ["jes"+s for s in self.jes_sources]

        # register functions
        event.register_function(
            event, 'Jet_ptResolution', partial(jet_pt_res, jers=self.jers),
        )
        event.register_function(
            event, 'Jet_jerSF', partial(
                jer_corr, jersfs=self.jersfs,
                maxdr_jets_with_genjets=self.maxdr_jets_with_genjets,
                ndpt_jets_with_genjets=self.ndpt_jets_with_genjets,
            ),
        )
        event.register_function(
            event, 'Jet_jesSF', partial(jes_corr, jesuncs=self.jesuncs),
        )

    def event(self, event):
        event.Jet_ptJESOnly = event.Jet_pt[:,:]
        event.MET_ptJESOnly = event.MET_pt[:]
        event.MET_phiJESOnly = event.MET_phi[:]
        event.MET_sumEtJESOnly = event.MET_sumEt[:]

        if self.apply_jer_corrections:
            sf = event.Jet_jerSF(event, "", 0.)
            event.Jet_pt = (event.Jet_ptJESOnly*sf)[:,:].astype(np.float32)

            met, mephi = met_shift(event)
            event.MET_pt = met[:].astype(np.float32)
            event.MET_phi = mephi[:].astype(np.float32)

            met_sumet = met_sumet_shift(event)
            event.MET_sumEt = met_sumet[:].astype(np.float32)

def jet_pt_res(ev, jers):
    indices = get_bin_indices(
        [np.abs(ev.Jet_eta.content),
         event_to_object_var(ev.fixedGridRhoFastjetAll, ev.Jet_ptJESOnly.starts, ev.Jet_ptJESOnly.stops)],
        [jers["eta_low"].values, jers["rho_low"].values],
        [jers["eta_high"].values, jers["rho_high"].values],
        1,
    )[:,0]
    df = jers.iloc[indices]
    params = df[["param0", "param1", "param2", "param3"]].values.astype(np.float32)
    ptbounds = df[["pt_low", "pt_high"]].values
    return awk.JaggedArray(
        ev.Jet_ptJESOnly.starts, ev.Jet_ptJESOnly.stops,
        jer_formula(
            np.minimum(np.maximum(ev.Jet_ptJESOnly.content, ptbounds[:,0]), ptbounds[:,1]).astype(np.float32),
            params[:,0], params[:,1], params[:,2], params[:,3],
        ),
    )

def jer_corr(ev, source, nsig, jersfs, maxdr_jets_with_genjets, ndpt_jets_with_genjets):
    flavour = "jerSF"
    if source == "jerSF" and nsig != 0.:
        updown = "Up" if nsig>0. else "Down"
        flavour += updown

    if not ev.hasbranch("Jet_JEC{}".format(flavour)):
        indices = get_bin_indices(
            [ev.Jet_eta.content],
            [jersfs["eta_low"].values],
            [jersfs["eta_high"].values],
            1,
        )[:,0]
        ressfs = jersfs.iloc[indices][["corr", "corr_up", "corr_down"]].values
        cjer = np.ones_like(ev.Jet_ptJESOnly.content, dtype=np.float32)
        cjer_up = np.ones_like(ev.Jet_ptJESOnly.content, dtype=np.float32)
        cjer_down = np.ones_like(ev.Jet_ptJESOnly.content, dtype=np.float32)

        # match gen jets
        gidx = match_jets_from_genjets(
            ev, maxdr_jets_with_genjets, ndpt_jets_with_genjets,
        )
        mask = (gidx>=0)
        indices = (ev.GenJet_pt.starts+gidx[mask]).content
        gpt_matched = ev.GenJet_pt.content[indices]
        mask = mask.content

        gen_var = np.abs(1.-gpt_matched/ev.Jet_ptJESOnly.content[mask])
        gaus_var = np.random.normal(0., gen_var)
        cjer[mask] = 1. + (ressfs[mask,0]-1.)*gaus_var
        cjer_up[mask] = 1. + (ressfs[mask,1]-1.)*gaus_var
        cjer_down[mask] = 1. + (ressfs[mask,2]-1.)*gaus_var

        # unmatched gen jets
        gaus_var = np.random.normal(0., ev.Jet_ptResolution(ev).content[~mask])
        ressfs_mod = ressfs[~mask]**2-1.
        ressfs_mod[ressfs_mod<0.] = 0.
        cjer[~mask] = 1. + gaus_var*np.sqrt(ressfs_mod[:,0])
        cjer_up[~mask] = 1. + gaus_var*np.sqrt(ressfs_mod[:,1])
        cjer_down[~mask] = 1. + gaus_var*np.sqrt(ressfs_mod[:,2])

        # negative checks
        cjer[cjer<0.] = 0.
        cjer_up[cjer_up<0.] = 0.
        cjer_down[cjer_down<0.] = 0.

        cjer_up[cjer>0.] = (cjer_up/cjer-1.)[cjer>0.]
        cjer_up[cjer==0.] = 0.
        cjer_down[cjer>0.] = (cjer_down/cjer-1.)[cjer>0.]
        cjer_down[cjer==0.] = 0.

        # write to event
        starts, stops = ev.Jet_ptJESOnly.starts, ev.Jet_ptJESOnly.stops
        ev.Jet_JECjerSF = awk.JaggedArray(starts, stops, cjer)
        ev.Jet_JECjerSFUp = awk.JaggedArray(starts, stops, cjer_up)
        ev.Jet_JECjerSFDown = awk.JaggedArray(starts, stops, cjer_down)

    return getattr(ev, "Jet_JEC{}".format(flavour))

def jes_corr(ev, source, nsig, jesuncs):
    flavour = source
    if source in ev.JesSources and nsig != 0.:
        updown = "Up" if nsig>0. else "Down"
        flavour += updown
    else:
        starts = ev.Jet_pt.starts
        stops = ev.Jet_pt.stops
        return awk.JaggedArray(
            starts, stops, np.ones_like(ev.Jet_pt.content, dtype=np.float32),
        )

    if not ev.hasbranch("Jet_JEC{}".format(flavour)):
        df = jesuncs[jesuncs["source"]==(source[3:] if source.startswith("jes") else source)]

        indices = get_bin_indices(
            [ev.Jet_eta.content], [df["eta_low"].values],
            [df["eta_high"].values], 1,
        )[:,0]

        pt = np.array(list(df.iloc[indices]["pt"].values))
        corr_up = np.array(list(df.iloc[indices]["corr_up"].values))
        corr_down = np.array(list(df.iloc[indices]["corr_down"].values))

        corr_up = interpolate(ev.Jet_ptJESOnly.content, pt, corr_up).astype(np.float32)
        corr_down = interpolate(ev.Jet_ptJESOnly.content, pt, corr_down).astype(np.float32)

        starts = ev.Jet_eta.starts
        stops = ev.Jet_eta.stops

        setattr(ev, "Jet_JEC{}Up".format(source), awk.JaggedArray(
            starts, stops, corr_up,
        ))
        setattr(ev, "Jet_JEC{}Down".format(source), awk.JaggedArray(
            starts, stops, -1.*corr_down,
        ))

    return getattr(ev, "Jet_JEC{}".format(flavour))

def read_table(path, underflow_cols=[], overflow_cols=[], csv=False):
    if not csv:
        df = pd.read_csv(path, sep='\s+')
    else:
        df = pd.read_csv(path, sep=',')

    for c in underflow_cols:
        df.loc[df[c]==df[c].min(), c] = -np.inf
    for c in overflow_cols:
        df.loc[df[c]==df[c].max(), c] = np.inf
    return df

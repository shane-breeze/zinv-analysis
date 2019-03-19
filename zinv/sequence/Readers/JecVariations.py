import numpy as np
import numba as nb
import pandas as pd
import awkward as awk
import re

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
            event.Jet_ptResolution.content,
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

        # Regex the variations
        comp_jes_regex = re.compile(self.jes_regex)
        variations = []
        for v in event.attribute_variation_sources:
            match = comp_jes_regex.search(v)
            if match:
                vari = match.group("source")
                if vari not in variations:
                    variations.append(vari)

        self.jesuncs = self.jesuncs.loc[
            self.jesuncs["source"].isin(variations)
        ]
        self.jesuncs["pt"] = self.jesuncs["pt"].apply(lambda x: list(eval(x)))
        self.jesuncs["corr_up"] = self.jesuncs["corr_up"].apply(lambda x: list(eval(x)))
        self.jesuncs["corr_down"] = self.jesuncs["corr_down"].apply(lambda x: list(eval(x)))
        self.jes_sources = self.jesuncs["source"].unique()
        event.JetSources = self.jes_sources

    def event(self, event):
        self.do_jet_pt_resolution(event)
        self.do_jer_correction(event)
        for source in self.jes_sources:
            self.do_jes_correction(event, source)

    def do_jet_pt_resolution(self, event):
        indices = get_bin_indices(
            [event.Jet_eta.content,
             event_to_object_var(event.fixedGridRhoFastjetAll, event.Jet_pt.starts, event.Jet_pt.stops)],
            [self.jers["eta_low"].values, self.jers["rho_low"].values],
            [self.jers["eta_high"].values, self.jers["rho_high"].values],
            1,
        )[:,0]
        df = self.jers.iloc[indices]
        params = df[["param0", "param1", "param2", "param3"]].values.astype(np.float32)
        ptbounds = df[["pt_low", "pt_high"]].values
        event.Jet_ptResolution = awk.JaggedArray(
            event.Jet_pt.starts, event.Jet_pt.stops,
            jer_formula(
                np.minimum(np.maximum(event.Jet_pt.content, ptbounds[:,0]), ptbounds[:,1]).astype(np.float32),
                params[:,0], params[:,1], params[:,2], params[:,3],
            ),
        )

    def do_jer_correction(self, event):
        indices = get_bin_indices(
            [event.Jet_eta.content],
            [self.jersfs["eta_low"].values],
            [self.jersfs["eta_high"].values],
            1,
        )[:,0]
        ressfs = self.jersfs.iloc[indices][["corr", "corr_up", "corr_down"]].values
        cjer = np.ones_like(event.Jet_pt.content, dtype=np.float32)
        cjer_up = np.ones_like(event.Jet_pt.content, dtype=np.float32)
        cjer_down = np.ones_like(event.Jet_pt.content, dtype=np.float32)

        # match gen jets
        gidx = match_jets_from_genjets(
            event, self.maxdr_jets_with_genjets, self.ndpt_jets_with_genjets,
        )
        event.Jet_genJetIdx = gidx
        event.Jet_sjer = awk.JaggedArray(
            event.Jet.pt.starts, event.Jet.pt.stops, ressfs[:,0],
        )
        mask = (gidx>=0)
        indices = (event.GenJet_pt.starts+gidx[mask]).content
        gpt_matched = event.GenJet_pt.content[indices]
        mask = mask.content

        gen_var = np.abs(1.-gpt_matched/event.Jet_pt.content[mask])
        gaus_var = np.random.normal(0., gen_var)
        cjer[mask] = 1. + (ressfs[mask,0]-1.)*gaus_var
        cjer_up[mask] = 1. + (ressfs[mask,1]-1.)*gaus_var
        cjer_down[mask] = 1. + (ressfs[mask,2]-1.)*gaus_var

        # unmatched gen jets
        gaus_var = np.random.normal(0., event.Jet_ptResolution.content[~mask])
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
        starts, stops = event.Jet_pt.starts, event.Jet_pt.stops
        jet_cjer = awk.JaggedArray(starts, stops, cjer)
        event.Jet_JECjerSFUp = awk.JaggedArray(starts, stops, cjer_up)
        event.Jet_JECjerSFDown = awk.JaggedArray(starts, stops, cjer_down)
        if self.apply_jer_corrections:
            event.Jet_ptJESOnly = event.Jet_pt[:,:]
            event.MET_ptJESOnly = event.MET_pt[:]
            event.MET_phiJESOnly = event.MET_phi[:]

            event.Jet_pt = (event.Jet_pt*jet_cjer)[:,:]
            met, mephi = met_shift(event)
            event.MET_pt = met[:]
            event.MET_phi = mephi[:]

    def do_jes_correction(self, event, source):
        df = self.jesuncs[self.jesuncs["source"]==source]

        indices = get_bin_indices(
            [event.Jet_eta.content],
            [df["eta_low"].values],
            [df["eta_high"].values],
            1,
        )[:,0]

        pt = np.array(list(df.iloc[indices]["pt"].values))
        corr_up = np.array(list(df.iloc[indices]["corr_up"].values))
        corr_down = np.array(list(df.iloc[indices]["corr_down"].values))

        corr_up = interpolate(event.Jet_ptJESOnly.content, pt, corr_up).astype(np.float32)
        corr_down = interpolate(event.Jet_ptJESOnly.content, pt, corr_down).astype(np.float32)

        starts = event.Jet_eta.starts
        stops = event.Jet_eta.stops

        setattr(event, "Jet_JECjes{}Up".format(source), awk.JaggedArray(
            starts, stops, corr_up,
        ))
        setattr(event, "Jet_JECjes{}Down".format(source), awk.JaggedArray(
            starts, stops, -1.*corr_down,
        ))

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

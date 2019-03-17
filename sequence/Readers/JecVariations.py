import numpy as np
import numba as nb
import pandas as pd
import awkward as awk
import re

from utils.NumbaFuncs import get_bin_indices, event_to_object_var, interpolate
from utils.Geometry import RadToCart2D, CartToRad2D

@nb.vectorize([nb.float32(nb.float32,nb.float32,nb.float32,nb.float32,nb.float32),
               nb.float64(nb.float64,nb.float64,nb.float64,nb.float64,nb.float64)])
def jer_formula(x, p0, p1, p2, p3):
    return np.sqrt(p0*np.abs(p0)/(x*x)+p1*p1*np.power(x,p3)+p2*p2)

def met_shift(ev, unclust_energy):
    @nb.njit
    def met_shift_numba(met, mephi, jpt, jptshift, jphi, jstarts, jstops):
        jpx_old, jpy_old = RadToCart2D(jpt, jphi)
        jpx_new, jpy_new = RadToCart2D(jptshift, jphi)

        mex, mey = RadToCart2D(met[:], mephi[:])
        for iev, (start, stop) in enumerate(zip(jstarts, jstops)):
            for iob in range(start, stop):
                if jpt[iob] > unclust_energy:
                    mex[iev] += jpx_old[iob]
                    mey[iev] += jpy_old[iob]
                if jptshift[iob]  > unclust_energy:
                    mex[iev] -= jpx_new[iob]
                    mey[iev] -= jpy_new[iob]

        return CartToRad2D(mex, mey)
    return met_shift_numba(
        ev.MET_ptJESOnly, ev.MET_phiJESOnly, ev.Jet_ptJESOnly.content,
        ev.Jet_pt.content, ev.Jet_phi.content,
        ev.Jet_pt.starts, ev.Jet_pt.stops,
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
        params = df[["param0", "param1", "param2", "param3"]].values
        ptbounds = df[["pt_low", "pt_high"]].values
        event.Jet_ptResolution = awk.JaggedArray(
            event.Jet_pt.starts, event.Jet_pt.stops,
            jer_formula(
                np.minimum(np.maximum(event.Jet_pt.content, ptbounds[:,0]), ptbounds[:,1]),
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
        jersfs = np.ones_like(event.Jet_pt.content, dtype=np.float32)
        jersfs_up = np.ones_like(event.Jet_pt.content, dtype=np.float32)
        jersfs_down = np.ones_like(event.Jet_pt.content, dtype=np.float32)

        # match gen jets
        gidx = event.Jet_genJetIdx
        gsize = event.GenJet_pt.counts
        mask = (gidx>=0) & (gidx<gsize)
        indices = (event.GenJet_pt.starts+gidx[mask]).content
        gpt_matched = event.GenJet_pt.content[indices]

        gen_var = 1.-gpt_matched/event.Jet_pt[mask].content
        jersfs[mask.content] = 1. + (ressfs[mask.content,0]-1.)*gen_var
        jersfs_up[mask.content] = 1. + (ressfs[mask.content,1]-1.)*gen_var
        jersfs_down[mask.content] = 1. + (ressfs[mask.content,2]-1.)*gen_var

        # unmatched gen jets
        gaus_var = np.random.normal(0., event.Jet_ptResolution[gidx<0].content)
        ressfs_mod = ressfs[(gidx<0).content]**2-1.
        ressfs_mod[ressfs_mod<0.] = 0.
        jersfs[(gidx<0).content] = 1. + gaus_var*np.sqrt(ressfs_mod[:,0])
        jersfs_up[(gidx<0).content] = 1. + gaus_var*np.sqrt(ressfs_mod[:,1])
        jersfs_down[(gidx<0).content] = 1. + gaus_var*np.sqrt(ressfs_mod[:,2])

        # negative checks
        jersfs[jersfs<0.] = 0.
        jersfs_up[jersfs_up<0.] = 0.
        jersfs_down[jersfs_down<0.] = 0.

        # write to event
        starts, stops = event.Jet_pt.starts, event.Jet_pt.stops
        event.Jet_JECjerSF = awk.JaggedArray(starts, stops, jersfs)
        event.Jet_JECjerSFUp = awk.JaggedArray(
            starts, stops,
            np.divide(jersfs_up, jersfs, out=np.ones_like(jersfs), where=(jersfs!=0.))-1.,
        )
        event.Jet_JECjerSFDown = awk.JaggedArray(
            starts, stops,
            np.divide(jersfs_down, jersfs, out=np.ones_like(jersfs), where=(jersfs!=0.))-1.,
        )
        if self.apply_jer_corrections:
            event.Jet_ptJESOnly = event.Jet_pt[:,:]
            event.MET_ptJESOnly = event.MET_pt[:]
            event.MET_phiJESOnly = event.MET_phi[:]

            event.Jet_pt = (event.Jet_pt*event.Jet_JECjerSF)[:,:]
            met, mephi = met_shift(event, self.unclust_threshold)
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

        corr_up = interpolate(event.Jet_ptJESOnly.content, pt, corr_up)
        corr_down = interpolate(event.Jet_ptJESOnly.content, pt, corr_down)

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

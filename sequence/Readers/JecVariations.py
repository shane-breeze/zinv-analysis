import awkward
import numpy as np
from numba import njit, int32, float32
from utils.Geometry import DeltaR2, RadToCart2D, CartToRad2D
import re

from collections import OrderedDict as odict

np.random.seed(123456)
regex = re.compile("jes(?P<source>.*)(Up|Down)")

class JecVariations(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.unclust_threshold = 15.

    def begin(self, event):
        self.isdata = event.config.dataset.isdata

        self.sources = list(set([
            v[:-2] if v.endswith("Up") else v[:-4] if v.endswith("Down") else v for v in event.variations
            if "jes" in v
        ]))

        self.variations = [("", [0.]*(len(self.sources)+2))]
        if self.do_jes:
            for idx, source in enumerate(self.sources):
                self.variations.extend([
                    (source+"Up",   [0.]*idx + [ 1.] + [0.]*(len(self.sources)-idx-1) + [0., 0.]),
                    (source+"Down", [0.]*idx + [-1.] + [0.]*(len(self.sources)-idx-1) + [0., 0.]),
                ])
        if self.do_jer and any("jer" in v for v in event.variations):
            self.variations.extend([
                ("jerUp",   [0.]*len(self.sources) + [ 1., 0.]),
                ("jerDown", [0.]*len(self.sources) + [-1., 0.]),
            ])
        if self.do_unclust and any("unclust" in v for v in event.variations):
            self.variations.extend([
                ("unclustUp",   [0.]*len(self.sources) + [0., 1.]),
                ("unclustDown", [0.]*len(self.sources) + [0.,-1.]),
            ])

        self.jesuncs = read_jesunc_file(self.jes_unc_file, self.sources, overflow=True)
        self.jersfs = read_jersf_file(self.jer_sf_file, overflow=True)
        self.jers = read_jer_file(self.jer_file, overflow=True)

    def event(self, event):
        if self.isdata:
            return True

        # Get jet pT resolution:
        event.Jet_ptResolution = get_jet_ptresolution(
            self.jers, event.Jet, event.fixedGridRhoFastjetAll,
        )

        # Do Jet-GenJet matching
        event.Jet_genJetMatchIdx = match_jets_genjets(
            event.Jet, event.GenJet,
        )

        # Get JER correction
        jersf, jersf_up, jersf_down = get_jer_sfs(self.jersfs, event.Jet)
        cjer, cjer_up, cjer_down = get_jer_correction(
            jersf, jersf_up, jersf_down, event.Jet, event.GenJet,
        )
        delta_jerup = cjer_up - cjer
        delta_jerdown = cjer - cjer_down
        if not self.apply_jer_corrections:
            cjer = np.ones(cjer.shape)
        event.Jet_jerCorrection = awkward.JaggedArray(
            event.Jet.starts, event.Jet.stops, cjer,
        )

        # Get delta unclustered energy x and y - absolute values
        delta_unclustx = event.MET_MetUnclustEnUpDeltaX
        delta_unclusty = event.MET_MetUnclustEnUpDeltaY

        results = []
        for key, vars in self.variations:
            match = regex.search(key)
            if match:
                source = match.group("source")
            else:
                source = key.replace("Up", "").replace("Down", "")

            if source in ["", "jer", "unclust"]:
                delta_jesup, delta_jesdown = 0., 0.
                jes_var = 0.
            else:
                # Get delta JES up and down - relative values
                delta_jesup, delta_jesdown = get_jes_sfs(self.jesuncs["jes"+source], event.Jet)
                sidx = self.sources.index("jes"+source)
                jes_var = vars[sidx]

            # jes at [0, N-2). jer at N-2. unclust at N-1
            jer_var = vars[-2]
            unclust_var = vars[-1]
            djes = delta_jesup if jes_var>=0. else delta_jesdown
            djer = delta_jerup if jer_var>=0. else delta_jerdown

            # jer_var and jes_var are additive
            (jets_pt, jets_mass), (met_pt, met_phi) = calculate_new_jets_met(
                jes_var*djes + jer_var*djer,
                unclust_var*delta_unclustx, unclust_var*delta_unclusty,
                self.unclust_threshold,
                event.Jet, event.MET,
            )
            results.append((
                key, (jets_pt, jets_mass, met_pt, met_phi),
            ))

        # Update nominal values AFTER all variations are calculated (they are
        # assumed with no JER correction applied)
        for key, (jets_pt, jets_mass, met_pt, met_phi) in results:
            setattr(event, "Jet_pt{}"  .format(key), jets_pt)
            setattr(event, "Jet_mass{}".format(key), jets_mass)
            setattr(event, "MET_pt{}"  .format(key), met_pt)
            setattr(event, "MET_phi{}" .format(key), met_phi)

        event.delete_branches([
            "Jet_ptResolution",
            "Jet_genJetMatchIdx",
            "Jet_jerCorrection",
        ])

################################################################################
def get_jet_ptresolution(jers, jets, rho):
    """Function to modify arguments that are sent to a numba-jitted function"""
    return awkward.JaggedArray(
        jets.starts, jets.stops,
        jit_get_jet_ptresolution(
            jers["bins"], jers["var_range"], jers["params"],
            jets.pt.content, jets.eta.content, jets.starts, jets.stops,
            rho,
        ),
    )

@njit
def jit_get_jet_ptresolution(bins, var_ranges, params,
                             jets_pt, jets_eta, jets_starts, jets_stops,
                             rho):
    resolution = np.zeros(jets_pt.shape[0], dtype=float32)
    for iev, (jb, je) in enumerate(zip(jets_starts, jets_stops)):
        for ij in range(jb, je):
            for ib in range(bins.shape[0]):
                within_eta = bins[ib][0] <= jets_eta[ij] < bins[ib][1]
                within_rho = bins[ib][2] <= rho[iev] < bins[ib][3]
                if within_eta and within_rho:
                    var_range = var_ranges[ib,:]
                    param = params[ib,:]

                    jet_pt = min(var_range[1], max(var_range[0], jets_pt[ij]))
                    resolution[ij] = np.sqrt(max(0.,
                        (param[0]*np.abs(param[0])/jet_pt**2) \
                        + param[1]**2*np.power(jet_pt, param[3]) \
                        + param[2]**2
                    ))
                    break
    return resolution

################################################################################
def match_jets_genjets(jets, genjets):
    """Function to modify arguments that are sent to a numba-jitted function"""
    return awkward.JaggedArray(
        jets.starts, jets.stops,
        jit_match_jets_genjets(
            jets.pt.content, jets.eta.content,
            jets.phi.content, jets.ptResolution.content,
            jets.starts, jets.stops,
            genjets.pt.content, genjets.eta.content, genjets.phi.content,
            genjets.starts, genjets.stops,
        ),
    )

@njit
def jit_match_jets_genjets(jets_pt, jets_eta, jets_phi, jets_res, jets_starts, jets_stops,
                           genjets_pt, genjets_eta, genjets_phi, genjets_starts, genjets_stops):
    match_idx = -1*np.ones(jets_pt.shape[0], dtype=int32)
    for iev, (j_b, j_e, gj_b, gj_e) in enumerate(zip(jets_starts, jets_stops, genjets_starts, genjets_stops)):
        for ijs in range(j_b, j_e):
            for igjs in range(gj_b, gj_e):
                dr2 = DeltaR2(jets_eta[ijs]-genjets_eta[igjs],
                              jets_phi[ijs]-genjets_phi[igjs])
                within_dpt = np.abs(jets_pt[ijs]-genjets_pt[igjs]) < 3.*jets_res[ijs]*jets_pt[ijs]
                if dr2 < 0.04 and within_dpt:
                    match_idx[ijs] = igjs-gj_b
                    break
    return match_idx

################################################################################
def get_jer_sfs(jersfs, jets):
    """Function to modify arguments that are sent to a numba-jitted function"""
    return jit_get_jer_sfs(
        jersfs["bins"], jersfs["corrs"], jersfs["corrs_up"], jersfs["corrs_down"],
        jets.eta.content,
    )
@njit
def jit_get_jer_sfs(bins, corrs, corrs_up, corrs_down, jets_eta):
    sfs = np.ones(jets_eta.shape[0], dtype=float32)
    sfs_up = np.ones(jets_eta.shape[0], dtype=float32)
    sfs_down = np.ones(jets_eta.shape[0], dtype=float32)
    for ij in range(jets_eta.shape[0]):
        for ib in range(bins.shape[0]):
            within_eta = bins[ib,0] <= jets_eta[ij] < bins[ib,1]
            if within_eta:
                sfs[ij] = corrs[ib]
                sfs_up[ij] = corrs_up[ib]
                sfs_down[ij] = corrs_down[ib]
                break
    return sfs, sfs_up, sfs_down

################################################################################
def get_jer_correction(jersf, jersf_up, jersf_down, jets, genjets):
    """Function to modify arguments that are sent to a numba-jitted function"""
    return jit_get_jer_correction(
        jersf, jersf_up, jersf_down,
        jets.pt.content, jets.genJetMatchIdx.content, jets.ptResolution.content,
        jets.starts, jets.stops,
        genjets.pt.content, genjets.starts, genjets.stops,
    )
@njit
def jit_get_jer_correction(jersf, jersf_up, jersf_down,
                           jets_pt, jets_genjetidx, jets_res, jets_starts, jets_stops,
                           genjets_pt, genjets_starts, genjets_stops):
    corrs = np.ones(jets_pt.shape[0], dtype=float32)
    corrs_up = np.ones(jets_pt.shape[0], dtype=float32)
    corrs_down = np.ones(jets_pt.shape[0], dtype=float32)
    for iev, (jb, je, gjb, gje) in enumerate(zip(jets_starts, jets_stops,
                                                 genjets_starts, genjets_stops)):
        for ij in range(jb, je):
            rel_genjetidx = jets_genjetidx[ij]
            if rel_genjetidx >= 0:
                gen_var = (jets_pt[ij]-genjets_pt[gjb+rel_genjetidx])/jets_pt[ij]
                corr = 1. + (jersf[ij]-1.)*gen_var
                corr_up = 1. + (jersf_up[ij]-1.)*gen_var
                corr_down = 1. + (jersf_down[ij]-1.)*gen_var
            else:
                #corr = np.random.lognormal(0., jets_res[ij])*np.sqrt(max(jersf[ij]**2-1., 0.)))
                gaus_var = np.random.normal(0., jets_res[ij])
                corr = 1. + gaus_var*np.sqrt(max(jersf[ij]**2-1., 0.))
                corr_up = 1. + gaus_var*np.sqrt(max(jersf_up[ij]**2-1., 0.))
                corr_down = 1. + gaus_var*np.sqrt(max(jersf_down[ij]**2-1., 0.))
            corrs[ij] = max(0., corr)
            corrs_up[ij] = max(0., corr_up)
            corrs_down[ij] = max(0., corr_down)
    return corrs, corrs_up, corrs_down

################################################################################
def get_jes_sfs(jesuncs, jets):
    return jit_get_jes_sfs(
        jesuncs["bins"], jesuncs["xvals"], jesuncs["yvals_up"], jesuncs["yvals_down"],
        jets.pt.content, jets.eta.content,
    )
@njit
def jit_get_jes_sfs(bins, xvals, yvals_up, yvals_down, jets_pt, jets_eta):
    sfs_up = np.ones(jets_pt.shape[0], dtype=float32)
    sfs_down = np.ones(jets_pt.shape[0], dtype=float32)
    for ij in range(jets_pt.shape[0]):
        for ib in range(bins.shape[0]):
            within_eta = bins[ib,0] <= jets_eta[ij] < bins[ib,1]
            if within_eta:
                sfs_up[ij] = interp(jets_pt[ij], xvals[ib], yvals_up[ib])
                sfs_down[ij] = interp(jets_pt[ij], xvals[ib], yvals_down[ib])
                break
    return sfs_up, sfs_down
@njit
def interp(x, xp, fp):
    nx = xp.shape[0]

    if x < xp[0]:
        return fp[0]
    elif x > xp[-1]:
        return fp[-1]

    for ix in range(nx-1):
        if xp[ix] <= x < xp[ix+1]:
            return (x - xp[ix]) * (fp[ix+1] - fp[ix]) / (xp[ix+1] - xp[ix]) + fp[ix]
    return np.nan

################################################################################
def calculate_new_jets_met(jes_var, unclustx_var, unclusty_var, unclust_threshold, jets, met):
    results = jit_calculate_new_jets_met(
        jes_var, unclustx_var, unclusty_var, unclust_threshold,
        jets.jerCorrection.content, jets.pt.content, jets.phi.content, jets.mass.content,
        jets.rawFactor.content, jets.starts, jets.stops,
        met.pt, met.phi,
    )
    return (
        (awkward.JaggedArray(jets.starts, jets.stops, r) for r in results[0]),
        results[1],
    )
@njit
def jit_calculate_new_jets_met(jes_var, unclustx_var, unclusty_var, unclust_threshold,
                               jets_jersf, jets_pt, jets_phi, jets_mass, jets_kraw, jets_starts, jets_stops,
                               met_pt, met_phi):
    # common jet correction factor
    jets_corr = (1 + jes_var)*jets_jersf

    # Modified jet pt and mass
    new_jets_pt = jets_corr*jets_pt
    new_jets_mass = jets_corr*jets_mass

    # Radian to cartesian for mex and mey corrections
    jets_px, jets_py = RadToCart2D(jets_pt, jets_phi)
    mex, mey = RadToCart2D(met_pt, met_phi)

    # Sum the jet modification contributions
    mex_jet_mod = np.zeros(met_pt.shape[0], dtype=float32)
    mey_jet_mod = np.zeros(met_pt.shape[0], dtype=float32)
    for iev, (jb, je) in enumerate(zip(jets_starts, jets_stops)):
        for ij in range(jb, je):
            # uncorrect JES
            if jets_pt[ij] > unclust_threshold:
                mex_jet_mod[iev] += jets_kraw[ij]*jets_px[ij]
                mey_jet_mod[iev] += jets_kraw[ij]*jets_py[ij]
            # correct JES and JER
            if new_jets_pt[ij] > unclust_threshold:
                mex_jet_mod[iev] -= (jets_corr[ij]+jets_kraw[ij]-1)*jets_px[ij]
                mey_jet_mod[iev] -= (jets_corr[ij]+jets_kraw[ij]-1)*jets_py[ij]

    new_mex = mex + mex_jet_mod + unclustx_var
    new_mey = mey + mey_jet_mod + unclusty_var

    new_met_pt, new_met_phi = CartToRad2D(new_mex, new_mey)
    return ((new_jets_pt, new_jets_mass), (new_met_pt, new_met_phi))

################################################################################
def read_jesunc_file(filename, sources, overflow=True):
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.read().splitlines()]

    lines = [l for l in lines if not l.startswith('#') and not l.startswith('{')]
    source_idx = odict([
        (l[1:-1], idx)
        for idx, l in enumerate(lines) if l.startswith('[')
    ])

    sources_lines = {}
    for source in sources:
        sidx = list(source_idx.values()).index(source_idx[source.replace("jes", "")])
        sidx_start = list(source_idx.values())[sidx]+1
        if sidx < len(source_idx):
            sidx_stop = list(source_idx.values())[sidx+1]
        else:
            sidx_stop = len(lines)+1
        source_lines = [l.split() for l in lines[sidx_start:sidx_stop]]

        bins = np.array([list(map(float, l[:2])) for l in source_lines])
        xvals = np.array([[float(l[3*(1+idx)]) for idx in range(50)] for l in source_lines])
        yvals_up = np.array([[float(l[3*(1+idx)+1]) for idx in range(50)] for l in source_lines])
        yvals_down = np.array([[float(l[3*(1+idx)+2]) for idx in range(50)] for l in source_lines])

        if overflow:
            bins[0,0] = -np.infty
            bins[-1,-1] = np.infty

        sources_lines[source] = {
            "bins": bins,
            "xvals": xvals,
            "yvals_up": yvals_up,
            "yvals_down": yvals_down,
        }
    return sources_lines

def read_jersf_file(filename, overflow=True):
    with open(filename, 'r') as f:
        lines = [l.strip().split() for l in f.read().splitlines()][1:]

    bins = np.array([list(map(float, l[:2])) for l in lines])
    corrs = np.array([float(l[3]) for l in lines])
    corrs_up = np.array([float(l[5]) for l in lines])
    corrs_down = np.array([float(l[4]) for l in lines])

    if overflow:
        bins[0,0] = -np.infty
        bins[-1,-1] = np.infty

    return {
        "bins": bins,
        "corrs": corrs,
        "corrs_up": corrs_up,
        "corrs_down": corrs_down,
    }

def read_jer_file(filename, overflow=True):
    with open(filename, 'r') as f:
        lines = [l.strip().split() for l in f.read().splitlines()][1:]

    bins = np.array([list(map(float, l[:4])) for l in lines])
    var_range = np.array([list(map(float, l[5:7])) for l in lines])
    params = np.array([list(map(float, l[7:11])) for l in lines])

    if overflow:
        bins[:,0][np.where(bins[:,0] == bins[:,0].min())] = -np.infty
        bins[:,1][np.where(bins[:,1] == bins[:,1].max())] = np.infty
        bins[:,2][np.where(bins[:,2] == bins[:,2].min())] = 0.
        bins[:,3][np.where(bins[:,3] == bins[:,3].max())] = np.infty

    return {
        "bins": bins,
        "var_range": var_range,
        "params": params,
    }

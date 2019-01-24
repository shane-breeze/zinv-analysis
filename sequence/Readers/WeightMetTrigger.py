import numpy as np
from scipy.special import erf
from numba import njit, float32

class WeightMetTrigger(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        self.cats = sorted([nmu for nmu in self.correction_files.keys()])
        self.bins = []
        self.corr = []
        self.stat_up = []
        self.stat_down = []
        self.syst_up = []
        self.syst_down = []
        self.eff = {}
        for nmuon in self.cats:
            bins, corr, stat_up, stat_down, syst_up, syst_down = read_file(self.correction_files[nmuon])
            self.bins.append(bins)
            self.corr.append(corr)
            self.stat_up.append(stat_up)
            self.stat_down.append(stat_down)
            self.syst_up.append(syst_up)
            self.syst_down.append(syst_down)

    def event(self, event):
        nmuons = event.MuonSelection.stops - event.MuonSelection.starts
        met = event.METnoX.pt
        corrs, stats_up, stats_down, systs_up, systs_down = get_correction(
            self.cats, self.bins, self.corr, self.stat_up, self.stat_down,
            self.syst_up, self.syst_down, nmuons, met,
        )
        event.Weight_MET *= corrs
        event.Weight_metTrigStatUp = np.divide(stats_up, corrs,
                                               out=np.ones_like(stats_up),
                                               where=corrs!=0)
        event.Weight_metTrigStatDown = np.divide(stats_down, corrs,
                                                 out=np.ones_like(stats_down),
                                                 where=corrs!=0)
        event.Weight_metTrigSystUp = np.divide(systs_up, corrs,
                                               out=np.ones_like(systs_up),
                                               where=corrs!=0)
        event.Weight_metTrigSystDown = np.divide(systs_down, corrs,
                                                 out=np.ones_like(systs_down),
                                                 where=corrs!=0)

@njit
def get_correction(cats, bins, incorr, instat_up, instat_down, insyst_up, insyst_down, nmuons, met):
    nev = nmuons.shape[0]
    corrs = np.ones(nev, dtype=float32)
    stats_up = np.ones(nev, dtype=float32)
    stats_down = np.ones(nev, dtype=float32)
    systs_up = np.ones(nev, dtype=float32)
    systs_down = np.ones(nev, dtype=float32)

    for iev in range(nev):
        if nmuons[iev] not in cats:
            continue

        cat = cats.index(nmuons[iev])
        xvals = (bins[cat][:,0]+bins[cat][:,1])/2
        corrs[iev] = interp(met[iev], xvals, incorr[cat])
        stats_up[iev] = interp(met[iev], xvals, instat_up[cat])
        stats_down[iev] = interp(met[iev], xvals, instat_down[cat])
        systs_up[iev] = interp(met[iev], xvals, insyst_up[cat])
        systs_down[iev] = interp(met[iev], xvals, insyst_down[cat])
    return corrs, stats_up, stats_down, systs_up, systs_down

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

def read_file(path):
    with open(path, 'r') as f:
        lines = [l.split()
                 for l in f.read().splitlines()
                 if l.strip()[0]!="#"][1:]

    bins = np.array([list(map(float, l[1:3])) for l in lines])
    corr = np.array([float(l[3]) for l in lines])
    stat_up = corr + np.array([float(l[5]) for l in lines])
    stat_down = corr - np.array([float(l[4]) for l in lines])
    syst_up = corr + np.array([float(l[7]) for l in lines])
    syst_down = corr + np.array([float(l[6]) for l in lines])

    #if overflow:
    #    bins[-1,-1] = np.infty

    return bins, corr, stat_up, stat_down, syst_up, syst_down

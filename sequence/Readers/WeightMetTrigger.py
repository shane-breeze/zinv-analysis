import numpy as np
import pandas as pd

from numba import njit, float32
from functools import partial
from utils.NumbaFuncs import interp, weight_numba
from cachetools.func import lru_cache

def evaluate_met_trigger(cats, xcents, params):
    @njit
    def met_trigger_numba(cats_, xcents_, incorr, nmuons, met):
        nev = met.shape[0]
        output = np.ones(nev, dtype=float32)
        for iev in range(nev):
            if nmuons[iev] not in cats_:
                continue
            cat = cats_.index(nmuons[iev])
            output[iev] = interp(met[iev], xcents_[cat], incorr[cat])
        return output

    @lru_cache(maxsize=32)
    def fevaluate_met_trigger(ev, evidx, nsig, source):
        nmuons = ev.MuonSelection(ev, 'pt').counts
        metnox = ev.METnoX_pt(ev)
        wmet = met_trigger_numba(cats, xcents, params[0], nmuons, metnox)

        if source == "metTrigStat":
            up = met_trigger_numba(cats, xcents, params[1], nmuons, metnox) / wmet - 1.
            down = met_trigger_numba(cats, xcents, params[2], nmuons, metnox) / wmet - 1.
        elif source == "metTrigSyst":
            up = met_trigger_numba(cats, xcents, params[3], nmuons, metnox) / wmet - 1.
            down = met_trigger_numba(cats, xcents, params[4], nmuons, metnox) / wmet - 1.
        else:
            up = np.zeros_like(wmet)
            down = np.zeros_like(wmet)

        return weight_numba(wmet, nsig, up, down)

    return lambda ev: fevaluate_met_trigger(ev, ev.iblock, ev.nsig, ev.source)

class WeightMetTrigger(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.cats = sorted([nmu for nmu in self.correction_files.keys()])
        self.xcents = []
        self.corr = []
        self.statup = []
        self.statdown = []
        self.systup = []
        self.systdown = []
        for nmuon in self.cats:
            df = read_file(self.correction_files[nmuon])
            self.xcents.append(df.eval("(x_low + x_high)/2.").values)
            self.corr.append(df["val"].values)
            self.statup.append(df["err_down"].values)
            self.statdown.append(df["err_up"].values)
            self.systup.append(df["syst_up"].values)
            self.systdown.append(df["syst_down"].values)

    def begin(self, event):
        params = (self.corr, self.statup, self.statdown, self.systup, self.systdown)
        event.WeightMETTrig = evaluate_met_trigger(self.cats, self.xcents, params)

def read_file(path):
    df = pd.read_table(path, sep='\s+')[[
        "x_low", "x_high", "val", "err_down", "err_up", "syst_up", "syst_down",
    ]]
    df["x_cent"] = df.eval("(x_low+x_high)/2.")
    return df

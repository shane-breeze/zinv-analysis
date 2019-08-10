import numpy as np
import numba as nb
import pandas as pd
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.NumbaFuncs import interp, weight_numba

def evaluate_met_trigger(ev, source, nsig, cats, xcents, params):
    @nb.njit
    def met_trigger_numba(cats_, xcents_, incorr, nmuons, met):
        nev = met.shape[0]
        output = np.ones(nev, dtype=np.float32)
        for iev in range(nev):
            if nmuons[iev] not in cats_:
                continue
            cat = cats_.index(nmuons[iev])
            output[iev] = interp(met[iev], xcents_[cat].astype(np.float32), incorr[cat].astype(np.float32))

        return output

    nmuons = ev.MuonSelection(ev, source, nsig, 'pt').counts
    metnox = ev.METnoX_pt(ev, source, nsig)
    wmet = met_trigger_numba(cats, xcents, params[0], nmuons, metnox)

    up = np.zeros_like(wmet)
    down = np.zeros_like(wmet)
    zero_mask = (wmet!=0.)

    if source == "metTrigStat":
        up[zero_mask] = met_trigger_numba(cats, xcents, params[1], nmuons, metnox)[zero_mask]/wmet[zero_mask] - 1.
        down[zero_mask] = met_trigger_numba(cats, xcents, params[2], nmuons, metnox)[zero_mask]/wmet[zero_mask] - 1.
    elif source == "metTrigSyst":
        up[zero_mask] = met_trigger_numba(cats, xcents, params[3], nmuons, metnox)[zero_mask]/wmet[zero_mask] - 1.
        down[zero_mask] = met_trigger_numba(cats, xcents, params[4], nmuons, metnox)[zero_mask]/wmet[zero_mask] - 1.

    return weight_numba(wmet, nsig, up, down)

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

        self.xcents = np.array(self.xcents, dtype=np.float32)
        self.corr = np.array(self.corr, dtype=np.float32)
        self.statup = np.array(self.statup, dtype=np.float32)
        self.statdown = np.array(self.statdown, dtype=np.float32)
        self.systup = np.array(self.systup, dtype=np.float32)
        self.systdown = np.array(self.systdown, dtype=np.float32)

    def begin(self, event):
        params = (
            self.corr,
            self.corr*(1.+self.statup),
            self.corr*(1.-self.statdown),
            self.corr*(1.+self.systup),
            self.corr*(1.-self.systdown),
        )
        event.register_function(
            event, "WeightMETTrig",
            partial(
                evaluate_met_trigger, cats=self.cats, xcents=self.xcents,
                params=params,
            ),
        )

def read_file(path):
    df = pd.read_csv(path, sep='\s+')[[
        "x_low", "x_high", "val", "err_down", "err_up", "syst_up", "syst_down",
    ]]
    df["x_cent"] = df.eval("(x_low+x_high)/2.")
    return df

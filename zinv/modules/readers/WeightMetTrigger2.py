import numpy as np
import numba as nb
import pandas as pd
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.NumbaFuncs import interp, weight_numba

def evaluate_met_trigger(ev, source, nsig, xcents, params, systs):
    @nb.njit
    def met_trigger_numba(xcents_, incorr, met):
        nev = met.shape[0]
        output = np.ones(nev, dtype=np.float32)
        for iev in range(nev):
            output[iev] = interp(
                met[iev], xcents_.astype(np.float32), incorr.astype(np.float32),
            )
        return output

    metnox = ev.METnoX_pt(ev, source, nsig)
    wmet = met_trigger_numba(xcents, params["value"], metnox)

    up = np.zeros_like(wmet)
    down = np.zeros_like(wmet)
    zero_mask = (wmet!=0.)

    for syst in systs:
        syst_name = "metTrig_{}Syst".format(syst if syst[0] == syst[0].lower() else syst[0].lower()+syst[1:])
        if source == syst_name:
            up[zero_mask] = met_trigger_numba(xcents, params[syst+"Up"], metnox)[zero_mask]/wmet[zero_mask] - 1.
            down[zero_mask] = met_trigger_numba(xcents, params[syst+"Down"], metnox)[zero_mask]/wmet[zero_mask] - 1.

    return weight_numba(wmet, nsig, up, down)

class WeightMetTrigger2(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        df = read_file(self.correction_file)

        # Assume all rows are the same for now
        systs = df["systs"].str.split("_").iloc[0]

        xcents = df["var0_cent"].values
        params = {"value": df["value"].values}

        for idx, syst in enumerate(systs):
            params[syst+"Up"] = params["value"]*(1.+df["syst{}_up".format(idx)].values)
            params[syst+"Down"] = params["value"]*(1.-df["syst{}_down".format(idx)].values)

        self.systs = systs
        self.xcents = xcents
        self.params = params

    def begin(self, event):
        event.register_function(
            event, "WeightMETTrig",
            partial(
                evaluate_met_trigger,
                xcents = self.xcents,
                params = self.params,
                systs = self.systs,
            ),
        )

def read_file(path):
    df = pd.read_csv(path)
    df = df[[
        "label", "vars", "systs", "var0_min", "var0_max", "value", "syst0_up",
        "syst0_down", "syst1_up", "syst1_down", "syst2_up", "syst2_down",
    ]]
    df["var0_cent"] = df.eval("(var0_min+var0_max)/2.")
    return df

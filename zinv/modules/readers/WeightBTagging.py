import numpy as np
import numba as nb
import pandas as pd
import awkward as awk
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.NumbaFuncs import weight_numba, get_bin_indices

dict_apply = np.vectorize(lambda d, x: d[x])

def btag_formula(x, df):
    @nb.njit(["float32[:](float32[:],int32[:],float32[:],float32[:],float32[:],float32[:],float32[:],float32[:],float32[:],float32[:],float32[:])"])
    def btag_formula_numba(x_, eqtype, xlow, xhigh, p0, p1, p2, p3, p4, p5, p6):
        xtest = np.minimum(np.maximum(x_, xlow), xhigh)
        sf = np.ones_like(xtest, dtype=np.float32)
        sf[eqtype==0] = (p0*((1+(p1*xtest))/(1+(p2*xtest))) + p3)[eqtype==0]
        sf[eqtype==1] = ((p0 + p1*xtest + p2*xtest**2 + p3*xtest**3)*(1 + (p4 + p5*xtest + p6*xtest**2)))[eqtype==1]
        sf[eqtype==2] = ((p0 + p1/(xtest**2) + p2*xtest)*(1 + (p4 + p5*xtest + p6*xtest**2)))[eqtype==2]
        return sf
    return btag_formula_numba(
        x, df["eqtype"].values.astype(np.int32),
        df["xlow"].values.astype(np.float32), df["xhigh"].values.astype(np.float32),
        df["p0"].values.astype(np.float32), df["p1"].values.astype(np.float32),
        df["p2"].values.astype(np.float32), df["p3"].values.astype(np.float32),
        df["p4"].values.astype(np.float32), df["p5"].values.astype(np.float32),
        df["p6"].values.astype(np.float32),
    )

def evaluate_btagsf(ev, source, nsig, df, attrs, h2f):
    jet_flavour = dict_apply(h2f, ev.Jet.hadronFlavour.content)

    # Create mask
    mask = np.ones((jet_flavour.shape[0], df.shape[0]), dtype=np.bool8)

    # Flavour mask
    event_attrs = [jet_flavour.astype(np.float32)]
    mins = [df["jetFlavor"].values.astype(np.float32)]
    maxs = [(df["jetFlavor"].values+1).astype(np.float32)]

    for jet_attr, df_attr in attrs:
        obj_attr = getattr(ev.Jet, jet_attr)
        if callable(obj_attr):
            obj_attr = obj_attr(ev, source, nsig)
        event_attrs.append(obj_attr.content.astype(np.float32))
        mins.append(df[df_attr+"Min"].values.astype(np.float32))
        maxs.append(df[df_attr+"Max"].values.astype(np.float32))

    # Create indices from mask
    indices = get_bin_indices(event_attrs, mins, maxs, 3)
    idx_central = indices[:,0]
    idx_down = indices[:,1]
    idx_up = indices[:,2]

    jpt = ev.Jet.ptShift(ev, source, nsig)
    sf = btag_formula(jpt.content, df.iloc[idx_central])
    sf_up = btag_formula(jpt.content, df.iloc[idx_up])
    sf_down = btag_formula(jpt.content, df.iloc[idx_down])

    sf_up = (source=="btagSF")*(sf_up/sf-1.)
    sf_down = (source=="btagSF")*(sf_down/sf-1.)
    return awk.JaggedArray(
        jpt.starts, jpt.stops, weight_numba(sf, nsig, sf_up, sf_down),
    )

class WeightBTagging(object):
    ops = {"loose": 0, "medium": 1, "tight": 2, "reshaping": 3}
    flavours = {"b": 0, "c": 1, "udsg": 2}
    hadron_to_flavour = {
        5: 0, -5: 0,
        4: 1, -4: 1,
        0: 2, 1: 2, 2: 2, 3: 2, -1: 2, -2: 2, -3: 2, 21: 2,
    }
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        df = pd.read_csv(self.calibration_file, sep=',\s+', engine='python')
        params = np.vstack(df["params"].apply(lambda x: eval(x[1:-1])))
        df["eqtype"] = params[:,0]
        df["xlow"] = params[:,1]
        df["xhigh"] = params[:,2]
        df["p0"] = params[:,3]
        df["p1"] = params[:,4]
        df["p2"] = params[:,5]
        df["p3"] = params[:,6]
        df["p4"] = params[:,7]
        df["p5"] = params[:,8]
        df["p6"] = params[:,9]

        op_num = self.ops[self.operating_point]
        df = df.loc[(df["CSVv2;OperatingPoint"] == op_num)]\
                .reset_index(drop=True)

        mask = np.zeros(df.shape[0], dtype=np.bool8)
        for flav, mtype in self.measurement_types.items():
            mask = mask | ((df["measurementType"]==mtype) & (df["jetFlavor"]==self.flavours[flav]))
        df = df.loc[mask]

        self.calibrations = df[[
            "sysType", "measurementType", "jetFlavor", "etaMin", "etaMax",
            "ptMin", "ptMax", "discrMin", "discrMax", "eqtype", "xlow", "xhigh",
            "p0", "p1", "p2", "p3", "p4", "p5", "p6",
        ]].sort_values(["sysType"]).reset_index(drop=True)

    def begin(self, event):
        attrs = [("eta", "eta"), ("ptShift", "pt")]
        if self.operating_point == "reshaping":
            attrs.append(("btagCSVV2", "discr"))

        event.register_function(
            event, "Jet_btagSF", partial(
                evaluate_btagsf, df=self.calibrations, attrs=attrs,
                h2f=self.hadron_to_flavour,
            ),
        )

    def end(self):
        self.calibrations = None

import numpy as np
import pandas as pd
import awkward as awk

from cachetools.func import lru_cache

from utils.Lambda import Lambda
from utils.NumbaFuncs import weight_numba, get_bin_mask, get_val_mask

dict_apply = np.vectorize(lambda d, x: d[x])
func_apply = np.vectorize(lambda f, x: f(x))

def evaluate_btagsf(df, attrs, h2f):
    @lru_cache(maxsize=32)
    def fevaluate_btagsf(ev, evidx, nsig, source, attrs_):
        jet_flavour = dict_apply(h2f, ev.Jet.hadronFlavour.content)

        # Create mask
        mask = np.ones((jet_flavour.shape[0], df.shape[0]), dtype=bool)

        # Flavour mask
        flav_bins = get_val_mask(jet_flavour, df["jetFlavor"].values)
        mask = mask & flav_bins

        for jet_attr, df_attr in attrs_:
            obj_attr = getattr(ev.Jet, jet_attr)
            if callable(obj_attr):
                obj_attr = obj_attr(ev)
            mask_attr = get_bin_mask(
                obj_attr.content,
                df[df_attr+"Min"].values,
                df[df_attr+"Max"].values,
            )
            mask = mask & mask_attr

        # Create indices from mask
        indices = np.array([np.nonzero(x)[0] for x in mask])
        idx_central = indices[:,0]
        idx_down = indices[:,1]
        idx_up = indices[:,2]

        jpt = ev.Jet.ptShift(ev)
        sf = func_apply(df.iloc[idx_central]["lambda_formula"].values, jpt.content)
        sf_up = func_apply(df.iloc[idx_up]["lambda_formula"].values, jpt.content)
        sf_down = func_apply(df.iloc[idx_down]["lambda_formula"].values, jpt.content)

        sf_up = (source=="btagSF")*(sf_up/sf-1.)
        sf_down = (source=="btagSF")*(sf_down/sf-1.)
        return awk.JaggedArray(
            jpt.starts, jpt.stops, weight_numba(sf, nsig, sf_up, sf_down),
        )

    return lambda ev: fevaluate_btagsf(ev, ev.iblock, ev.nsig, ev.source, tuple(attrs))

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

        df = pd.read_csv(self.calibration_file, sep=',\s*')
        df["formula"] = df["formula"].apply(lambda f: "x: "+f.replace('"', ''))

        op_num = self.ops[self.operating_point]
        df = df.loc[(df["CSVv2;OperatingPoint"] == op_num)]\
                .reset_index(drop=True)

        mask = np.zeros(df.shape[0], dtype=bool)
        for flav, mtype in self.measurement_types.items():
            mask = mask | ((df["measurementType"]==mtype) & (df["jetFlavor"]==self.flavours[flav]))
        df = df.loc[mask]

        self.calibrations = df[[
            "sysType", "measurementType", "jetFlavor", "etaMin", "etaMax",
            "ptMin", "ptMax", "discrMin", "discrMax", "formula",
        ]].sort_values(["sysType"]).reset_index(drop=True)

    def begin(self, event):
        self.calibrations["lambda_formula"] = self.calibrations["formula"].apply(Lambda)
        attrs = [("eta", "eta"), ("ptShift", "pt")]
        if self.operating_point == "reshaping":
            attrs.append(("btagCSVV2", "discr"))

        setattr(event, "Jet_btagSF", evaluate_btagsf(
            self.calibrations, attrs, self.hadron_to_flavour,
        ))

    def end(self):
        self.calibrations = None

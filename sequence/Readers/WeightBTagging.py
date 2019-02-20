import awkward
import numpy as np
import pandas as pd

from utils.Lambda import Lambda
from utils.NumbaFuncs import (get_bin_indices, get_val_indices,
                              get_bin_mask, get_val_mask)

dict_apply = np.vectorize(lambda d, x: d[x])

class WeightBTagging(object):
    ops = {"loose": 0, "medium": 1, "tight": 2, "reshaping": 3}
    flavours = {"b": 0, "c": 1, "udsg": 2}
    parton_to_flavour = {
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

        self.calibrations = df[["sysType", "measurementType", "jetFlavor",
                                "etaMin", "etaMax", "ptMin", "ptMax",
                                "discrMin", "discrMax", "formula"]]

    def begin(self, event):
        self.calibrations["lambda_formula"] = self.calibrations["formula"].apply(lambda f: Lambda(f))

    def end(self):
        self.calibrations = None

    def event(self, event):
        for syst, name in [("central", ""), ("up", "Up"), ("down", "Down")]:
            # systematic type
            df = self.calibrations.loc[self.calibrations["sysType"] == syst]\
                    .reset_index(drop=True)

            jet_flavour = dict_apply(
                self.parton_to_flavour,
                event.JetSelection.partonFlavour.content,
            )

            # Create mask
            mask = np.ones((jet_flavour.shape[0], df.shape[0]), dtype=bool)

            # exact values
            flav_bins = get_val_mask(jet_flavour, df["jetFlavor"].values)
            mask = mask & flav_bins

            attrs = [("eta", "eta"), ("pt", "pt")]
            if self.operating_point == "reshaping":
                attrs.append(("btagCSVV2", "discr"))

            for jet_attr, df_attr in attrs:
                obj_attr = getattr(event.JetSelection, jet_attr).content
                mask_attr = get_bin_mask(
                    obj_attr,
                    df[df_attr+"Min"].values,
                    df[df_attr+"Max"].values,
                )
                if jet_attr in ["pt"]:
                    mask_attr[obj_attr>=df[df_attr+"Max"].max()] = True
                mask = mask & mask_attr

            # Create indices from mask
            for x in mask:
                print(np.nonzero(x))
            indices = np.array([np.nonzero(x) for x in mask]).ravel()
            sf = np.vectorize(lambda f, x: f(x), otypes=[object])(
                df.iloc[indices]["lambda_formula"].values,
                event.JetSelection.pt.content,
            )

            setattr(event, "Jet_btagSF{}".format(name), awkward.JaggedArray(
                event.JetSelection.starts,
                event.JetSelection.stops,
                sf,
            ))

if __name__ == "__main__":
    weight_btagging = WeightBTagging(
        operating_point = "medium",
        measurement_types = {"b": "comb", "c": "comb", "udsg": "incl"},
        calibration_file = datapath+"/btagging/CSVv2_Moriond17_B_H.csv",
    )

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

        self.calibrations = df[["sysType", "measurementType", "jetFlavor",
                                "etaMin", "etaMax", "ptMin", "ptMax",
                                "discrMin", "discrMax", "formula"]]

    def begin(self, event):
        self.calibrations["lambda_formula"] = self.calibrations["formula"].apply(lambda f: Lambda(f))

    def end(self):
        self.calibrations = None

    def event(self, event):
        pt_out_of_bounds = np.zeros_like(event.Jet.pt.content, dtype=bool)
        for syst, name in [("central", ""), ("up", "Up"), ("down", "Down")]:
            # systematic type
            df = self.calibrations.loc[self.calibrations["sysType"] == syst]\
                    .reset_index(drop=True)

            jet_flavour = dict_apply(
                self.hadron_to_flavour,
                event.Jet.hadronFlavour.content,
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
                obj_attr = np.copy(getattr(event.Jet, jet_attr).content)
                if jet_attr in ["pt"]:
                    is_max_out_of_bounds = (obj_attr>=df[df_attr+"Max"].max())
                    is_min_out_of_bounds = (obj_attr<df[df_attr+"Min"].min())
                    pt_out_of_bounds = pt_out_of_bounds | is_max_out_of_bounds | is_min_out_of_bounds
                    obj_attr[is_max_out_of_bounds] = df[df_attr+"Max"].max()
                    obj_attr[is_min_out_of_bounds] = df[df_attr+"Min"].min()
                    pt_attr = obj_attr
                mask_attr = get_bin_mask(
                    obj_attr,
                    df[df_attr+"Min"].values,
                    df[df_attr+"Max"].values,
                )
                mask = mask & mask_attr

            # Create indices from mask
            indices = np.array([np.nonzero(x) for x in mask]).ravel()
            sf = np.ones_like(event.Jet.pt.content)
            eq = np.zeros_like(event.Jet.pt.content, dtype=object)
            indices_nonzero = np.vectorize(lambda x: x.shape[0]>0)(indices)
            eq[indices_nonzero] = df.iloc[indices[indices_nonzero]]["formula"].values
            sf[indices_nonzero] = np.vectorize(
                lambda f, x: f(x), otypes=[object],
            )(
                df.iloc[indices[indices_nonzero]]["lambda_formula"].values,
                pt_attr[indices_nonzero],
            )

            setattr(event, "Jet_btagSF{}".format(name), awkward.JaggedArray(
                event.Jet.starts,
                event.Jet.stops,
                sf,
            ))
            setattr(event, "Jet_btagEq{}".format(name), awkward.JaggedArray(
                event.Jet.starts,
                event.Jet.stops,
                eq,
            ))

        sf = event.Jet.btagSF.content
        sfup = event.Jet.btagSFUp.content
        sfdown = event.Jet.btagSFDown.content
        sfup[pt_out_of_bounds] = sf[pt_out_of_bounds]+2*(sfup[pt_out_of_bounds]-sf[pt_out_of_bounds])
        sfdown[pt_out_of_bounds] = sf[pt_out_of_bounds]+2*(sfdown[pt_out_of_bounds]-sf[pt_out_of_bounds])

        bjets_sf = event.JetSelection.btagSF[event.JetSelection.btagCSVV2>self.threshold]
        bjets_sfup = event.JetSelection.btagSFUp[event.JetSelection.btagCSVV2>self.threshold]
        bjets_sfdown = event.JetSelection.btagSFDown[event.JetSelection.btagCSVV2>self.threshold]

        event.Weight_btagVeto = (1.-bjets_sf).prod()
        event.Weight_btagVetoUp = (1.-bjets_sfup).prod() / event.Weight_btagVeto
        event.Weight_btagVetoDown = (1.-bjets_sfdown).prod() / event.Weight_btagVeto

        event.Weight_MET *= event.Weight_btagVeto
        event.Weight_SingleMuon *= event.Weight_btagVeto
        event.Weight_SingleElectron *= event.Weight_btagVeto

if __name__ == "__main__":
    weight_btagging = WeightBTagging(
        operating_point = "medium",
        threshold = 0.8484,
        measurement_types = {"b": "comb", "c": "comb", "udsg": "incl"},
        calibration_file = datapath+"/btagging/CSVv2_Moriond17_B_H.csv",
    )

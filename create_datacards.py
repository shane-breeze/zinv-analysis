import argparse
import copy
import numpy as np
from tabulate import tabulate as tab

try: import cPickle as pickle
except ImportError: import pickle

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.width', None)

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input pickle file")
    parser.add_argument("--binning", type=str, default="[200.]",
                        help="Binning to use")

    return parser.parse_args()

def open_df(path):
    with open(path, 'r') as f:
        df = pickle.load(f)
    df = df.reset_index("variable0", drop=True)
    return df

def reformat(df, binning):
    # Rebin
    def rebin(df, bins):
        df = df.reset_index(["bin0_low", "bin0_upp"])
        df["merge_idx"] = df["bin0_low"].apply(
            lambda b: next(idx for idx, nb in enumerate(bins[1:]) if b<nb)
        )
        df = df.drop(["bin0_low", "bin0_upp"], axis=1)
        df = df.set_index("merge_idx", append=True)
        df = df.groupby(df.index.names).sum()
        df = df.reset_index("merge_idx")
        df["bin0_low"] = df["merge_idx"].apply(lambda i: bins[i])
        df["bin0_upp"] = df["merge_idx"].apply(lambda i: bins[i+1])
        df = df.drop("merge_idx", axis=1)\
                .set_index("bin0_low", append=True)\
                .set_index("bin0_upp", append=True)
        return df
    #bins = [-np.infty]+list(np.linspace(200., 1000., 17))+[np.infty]
    bins = [-np.infty]+list(binning)+[np.infty]
    df = rebin(df, bins)

    # Rename regions
    def rename_level_values(df, level, name_map):
        levels = df.index.names
        df = df.reset_index(level)
        df[level] = df[level].map(name_map, na_action='ignore')
        df = df[~df[level].isna()]
        df = df.set_index(level, append=True)\
                .reorder_levels(levels)
        return df
    df = rename_level_values(df, "dataset", {"MET": "MET"})
    df = rename_level_values(df, "region", {
        "Monojet": "monojet", "SingleMuon": "singlemu", "DoubleMuon": "doublemu",
    })
    df = rename_level_values(df, "process", {
        "ZJetsToNuNu":    "znunu",
        "DYJetsToMuMu":   "zmumu",
        "WJetsToENu":     "wlnu",
        "WJetsToMuNu":    "wlnu",
        "WJetsToTauNu":   "wlnu",
        "QCD":            "qcd",
        "TTJets":         "bkg",
        "Diboson":        "bkg",
        "DYJetsToEE":     "bkg",
        "DYJetsToTauTau": "bkg",
        "EWKV2Jets":      "bkg",
        "SingleTop":      "bkg",
        "G1Jet":          "bkg",
        "VGamma":         "bkg",
        "MET":            "MET",
        "SingleMuon":     "SingleMuon",
        "SingleElectron": "SingleElectron",
    })
    df = df.groupby(df.index.names).sum()

    levels = df.index.names
    df = df.reset_index(["process", "dataset"])
    df["drop"] = df.apply(lambda row: row["process"] in ["MET", "SingleMuon", "SingleElectron"]\
                          and row["process"] != row["dataset"], axis=1)
    df = df.set_index(["process", "dataset"], append=True)\
            .reorder_levels(levels)
    df = df[~df["drop"]].drop("drop", axis=1)
    return df

def create_datacards(df):
    for category, dfgroup in df.groupby(["bin0_low", "bin0_upp"]):
        if np.isinf(category[0]):
            continue

        # data obs
        df_obs = dfgroup[dfgroup.index.get_level_values("process").isin(["MET"])]["yield"]
        df_obs = df_obs.reset_index([l for l in df.index.names if l != "region"], drop=True)
        dfgroup = dfgroup[~dfgroup.index.get_level_values("process").isin(["MET"])]

        # mc rate
        df_rate = dfgroup[dfgroup.index.get_level_values("weight").isin(["nominal"])]["yield"]
        df_rate = df_rate.reset_index([l for l in df.index.names if l not in ["region", "process"]], drop=True)
        df_rate = df_rate.fillna(1e-10)
        df_rate = df_rate.reset_index("process")
        df_rate["proc"] = df_rate["process"].map(
            dict(zip(*zip(*enumerate(["znunu", "zmumu", "wlnu", "qcd", "bkg"], 0))[::-1]))
        )
        df_rate = df_rate.set_index("process", append=True)
        df_rate[df_rate["yield"]<0.] = np.nan
        df_rate[df_rate.groupby("region").apply(lambda x: x/x.sum())["yield"]<0.001] = np.nan
        print(df_rate)
        exit()

        # nuisances
        df_nominal = dfgroup[dfgroup.index.get_level_values("weight").isin(["nominal"])]
        dfgroup = dfgroup[~dfgroup.index.get_level_values("weight").isin(["nominal"])]

        df_nuis = dfgroup.pivot_table(values='yield', index=['region', 'process'],
                                      columns=['weight'], aggfunc=np.sum)
        df_nominal = df_nominal.pivot_table(values='yield', index=['region', 'process'],
                                            columns=['weight'], aggfunc=np.sum)
        df_nuis = df_nuis.divide(df_nominal["nominal"], axis=0)

        df_nuis.loc[:,"lumiUp"] = 1.025
        df_nuis.loc[:,"lumiDown"] = 1/1.025
        df_nuis.loc[df_nuis.index.get_level_values("process").isin(["wlnu"]),"lumiUp"] = 1.
        df_nuis.loc[df_nuis.index.get_level_values("process").isin(["wlnu"]),"lumiDown"] = 1.

        # Fix for bug in zinv-analysis framework
        df_nuis.loc[:,"metTrigSFDown"] = 1/df_nuis[["metTrigSFDown"]]

        # Fix one-sided uncertainties (lack of stats in events which differ)
        nuisances = list(set(c.replace("Up","").replace("Down","") for c in df_nuis.columns))
        for n in nuisances:
            df_temp = df_nuis.loc[:, [n+"Up", n+"Down"]]
            df_temp.loc[((df_temp-1).prod(axis=1)>0) & (np.abs(df_temp[n+"Up"]-1)>=np.abs(df_temp[n+"Down"]-1)), n+"Down"] = 1.
            df_temp.loc[((df_temp-1).prod(axis=1)>0) & (np.abs(df_temp[n+"Up"]-1)<np.abs(df_temp[n+"Down"]-1)), n+"Up"] = 1.
            df_nuis[[n+"Up", n+"Down"]] = df_temp

        # symmetric uncertainties
        #for n in nuisances:
        #    df_nuis[n+"Up"] = np.sqrt(df_nuis[n+"Up"]/df_nuis[n+"Down"])
        #    df_nuis[n+"Down"] = 1/df_nuis[n+"Up"]

        # order
        order = [
            ("monojet", "znunu"), ("monojet", "zmumu"), ("monojet", "wlnu"),
            ("monojet", "qcd"), ("monojet", "bkg"),
            ("singlemu", "znunu"), ("singlemu", "zmumu"), ("singlemu", "wlnu"),
            ("singlemu", "qcd"), ("singlemu", "bkg"),
            ("doublemu", "znunu"), ("doublemu", "zmumu"), ("doublemu", "wlnu"),
            ("doublemu", "qcd"), ("doublemu", "bkg"),
        ]
        df_obs = df_obs.loc[list(set(o[0] for o in order))].dropna()
        df_rate = df_rate.reindex(order).dropna()
        df_nuis = df_nuis.loc[df_rate.index]

        # Remove muonTrig
        df_nuis = df_nuis[[c for c in df_nuis.columns if "muonTrig" not in c]]

        low = int(category[0]) if not np.isinf(category[0]) else "Inf"
        high = int(category[1]) if not np.isinf(category[1]) else "Inf"
        filename = "Zinv_METnoX-{}To{}.txt".format(low, high)

        create_datacard(df_obs, df_rate, df_nuis, filename)

def create_datacard(df_obs, df_rate, df_nuis, filename):
    # IDX
    dc = tab([
        ["imax * number of channels"],
        ["jmax * number of backgrounds"],
        ["kmax * number of nuisance parameters"],
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # SHAPES
    df_obs = df_obs.reset_index()
    dc += tab([
        ["shapes", "*", region, "FAKE"]
        for region in df_obs["region"]
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # OBS
    dc += tab([
        ["bin"] + list(df_obs["region"]),
        ["observation"] + map(int, list(df_obs["yield"])),
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # RATE
    df_rate = df_rate.reset_index()
    dc += tab([
        ["bin"] + list(df_rate["region"]),
        ["process"] + list(df_rate["process"]),
        ["process"] + map(int, list(df_rate["proc"])),
        ["rate"] + list(df_rate["yield"]),
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # NUISANCES
    nuisances = sorted(list(set(c.replace("Up", "").replace("Down", "") for c in df_nuis.columns)))

    nuisance_block = []
    for nuis in nuisances:
        nuisance_subblock = [nuis, "lnN"]
        for up, down in zip(df_nuis[nuis+"Up"], df_nuis[nuis+"Down"]):
            if np.isnan([up, down]).any():
                # non-number
                value = "-"
            else:
                # number
                if np.abs(up*down-1)<0.005:
                    # symmetric
                    mean = np.sqrt(up/down)
                    if np.abs(mean-1)<1e-5:
                        # zero
                        value = "-"
                    else:
                        # non-zero
                        value = str(mean)
                else:
                    # asymmetric
                    value = "{}/{}".format(up, down)
            nuisance_subblock.append(value)
        nuisance_block.append(nuisance_subblock)
    dc += tab(nuisance_block, [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    tf_wlnu_regions = list(df_rate[df_rate["process"]=="wlnu"]["region"])
    r_nunu_regions = list(df_rate[df_rate["process"]=="znunu"]["region"])
    r_mumu_regions = list(df_rate[df_rate["process"]=="zmumu"]["region"])

    # PARAMS
    dc += tab(
        [["tf_wlnu", "rateParam", region,  "wlnu",  1] for region in tf_wlnu_regions] +\
        [["r_z",     "rateParam", region,  "znunu", 1] for region in r_nunu_regions] +\
        [["r_z",     "rateParam", region, "zmumu", 1] for region in r_mumu_regions],
        [], tablefmt="plain",
    )

    with open(filename, 'w') as f:
        f.write(dc)
    logger.info("Created {}".format(filename))

def main():
    options = parse_args()

    df = open_df(options.input)
    df = reformat(df, eval(options.binning))
    create_datacards(df)

if __name__ == "__main__":
    main()

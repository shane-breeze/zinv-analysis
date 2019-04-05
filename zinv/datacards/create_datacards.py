import argparse
import copy
import numpy as np
import oyaml as yaml
import pandas as pd
from array import array
from tabulate import tabulate as tab

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

import ROOT
ROOT.gROOT.SetBatch(True)

try: import cPickle as pickle
except ImportError: import pickle

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input pickle file")
    parser.add_argument("config", type=str, help="Yaml config file")
    parser.add_argument("-s", "--systematics", type=int, default=1,
                        help="Turn systematics on(1)/off(0)")
    return parser.parse_args()

def open_df(cfg):
    path = cfg["input"]
    with open(path, 'r') as f:
        _, df = pickle.load(f)
        df = df.reset_index("variable0", drop=True)

    def rename_level_values(df, level, name_map):
        levels = df.index.names
        df = df.reset_index(level)
        mask = df[level].isin(name_map.keys())
        df.loc[mask, level] = df.loc[mask, level].map(name_map, na_action='ignore')
        df = df[~df[level].isna()]
        df = df.set_index(level, append=True).reorder_levels(levels)
        return df

    df = rename_level_values(df, "weight", {"": "nominal"})

    return df

def smooth(x, y, err):
    s = UnivariateSpline(x, y, w=1./err, k=2, s=x.shape[0]*4)
    return gaussian_filter1d(s(x), 1)

def process_syst(df, syst=None, how_up=lambda x: x.mean()+x.std(), how_down = lambda x: x.mean()-x.std()):
    df_nominal = df[df.index.get_level_values("weight").isin(["nominal"])]
    df_syst = df[df.index.get_level_values("weight").str.contains(syst)]
    #df_syst = df_syst[df_syst["yield"]>=0.]
    df_syst = df_syst.dropna()
    df_syst = df_syst.reset_index("weight", drop=True)
    all_indexes = df.index.names
    indexes = [ind for ind in df.index.names if ind not in ["weight"]]
    df_syst_up = df_syst.groupby(indexes).apply(how_up)
    df_syst_down = df_syst.groupby(indexes).apply(how_down)

    df_syst_up["weight"] = syst+"Up"
    df_syst_down["weight"] = syst+"Down"
    df_syst_up = df_syst_up.set_index("weight", append=True)\
            .reorder_levels(all_indexes)
    df_syst_down = df_syst_down.set_index("weight", append=True)\
            .reorder_levels(all_indexes)

    df = df[~df.index.get_level_values("weight").str.contains(syst)]
    df = pd.concat([df, df_syst_up, df_syst_down], axis=0).sort_index()
    return df

def apply_selection(df, selection):
    # Apply selection
    df["selection"] = df.eval(selection)
    df_s = df[df["selection"]==True]
    if "bin0" in selection:
        df_s = df_s.reset_index(["bin0_low", "bin0_upp"], drop=True)
    if "bin1" in selection:
        df_s = df_s.reset_index(["bin1_low", "bin1_upp"], drop=True)
    df_s = df_s.groupby(list(df_s.index.names)).sum()\
            .drop("selection", axis=1)
    return df_s

def reformat(df, cfg):
    # Rebin
    def rebin(df, bins, binvar):
        df = df.reset_index([binvar+"_low", binvar+"_upp"])
        df["merge_idx"] = df[binvar+"_low"].apply(
            lambda b: next(idx for idx, nb in enumerate(bins[1:]) if b<nb)
        )
        df = df.drop([binvar+"_low", binvar+"_upp"], axis=1)
        df = df.set_index("merge_idx", append=True)
        df = df.groupby(df.index.names).sum()
        df = df.reset_index("merge_idx")
        df[binvar+"_low"] = df["merge_idx"].apply(lambda i: bins[i])
        df[binvar+"_upp"] = df["merge_idx"].apply(lambda i: bins[i+1])
        df = df.drop("merge_idx", axis=1)\
                .set_index(binvar+"_low", append=True)\
                .set_index(binvar+"_upp", append=True)
        return df
    binvar, binning = cfg["binning"].split("=")
    binning = eval(binning)
    bins = [-np.infty]+list(binning)+[np.infty]
    df = rebin(df, bins, binvar)

    #if "lhePdf" in cfg["systematics"]:
    #    df = process_syst(
    #        df, syst="lhePdf",
    #        how_up=lambda x: (x**2).mean().sqrt()-x.mean(),
    #        how_down=lambda x: (x**2).mean().sqrt()-x.mean(),
    #    )

    # Rename
    dfs = []
    for key, val in cfg["conversions"].items():
        levels = df.index.names
        labs = val["labels"]

        sdf = df.reset_index(labs)
        sdf["drp"] = ""
        for i, l in enumerate(labs):
            sdf["drp"] += sdf[l]
            if i < len(labs)-1:
                sdf["drp"] += "__"

        for new_label in val["new_labels"]:
            rp_map = {"__".join(k): "__".join(new_label) for k in val["old_labels"]}
            ssdf = sdf.copy()
            ssdf["drp"] = ssdf["drp"].map(rp_map, na_action='ignore')
            ssdf = ssdf[~ssdf["drp"].isna()]
            for i, l in enumerate(labs):
                ssdf[l] = ssdf["drp"].apply(lambda x: x.split("__")[i])
            ssdf = ssdf.set_index(labs, append=True)\
                    .drop("drp", axis=1)\
                    .reorder_levels(levels)
            dfs.append(ssdf)
    dfs = pd.concat(dfs)
    dfs = dfs.groupby(dfs.index.names).sum()
    return dfs

def create_shape_datacards(df, cfg):
    binlab, binning = cfg["binning"].split("=")
    binning = eval(binning)
    all_inds = df.index.names
    all_inds_no_bins = [i for i in all_inds if "bin" not in i]

    # open ROOT file
    rfile = ROOT.TFile.Open("Zinv_METnoX-ShapeTemplates_{}.root".format(cfg["name"]), 'RECREATE')

    df = df.reset_index([i for i in all_inds if "bin" in i])

    # Smooth
    # Skip smoothing for now
    #for s in set([s[:-2] if s.endswith("Up") else s[:-4] if s.endswith("Down") else s for s in df.index.get_level_values("weight").unique()]):
    #    if s == "nominal":
    #        continue

    #    for (d, r, p, n), _ in df.groupby([c for c in all_inds_no_bins if c!="weight"]):
    #        if p == "data_obs":
    #            continue

    #        df_nominal = df.loc[(d, r, p, "nominal", n), :]
    #        try:
    #            df_up = df.loc[(d, r, p, s+"Up", n), :]
    #        except KeyError:
    #            print(d, r, p, s+"Up", n)
    #            continue
    #        try:
    #            df_down = df.loc[(d, r, p, s+"Down", n), :]
    #        except KeyError:
    #            print(d, r, p, s+"Down", n)
    #            continue
    #        xvals = (np.array(binning)[1:]+np.array(binning)[:-1])/2

    #        df_nominal = df_nominal.reset_index(drop=True)\
    #                .set_index("bin0_low")\
    #                .reindex(np.array(binning[:-1]))\
    #                .fillna(0.)\
    #                .reset_index("bin0_low")
    #        df_up = df_up.reset_index(drop=True)\
    #                .set_index("bin0_low")\
    #                .reindex(np.array(binning[:-1]))\
    #                .fillna(0.)\
    #                .reset_index("bin0_low")
    #        df_down = df_down.reset_index(drop=True)\
    #                .set_index("bin0_low")\
    #                .reindex(np.array(binning[:-1]))\
    #                .fillna(0.)\
    #                .reset_index("bin0_low")

    #        ratio = df_up["yield"].values/df_nominal["yield"].values
    #        ratio[np.isnan(ratio) | np.isinf(ratio)] = 1.
    #        ratio_err = np.sqrt(np.abs(df_up["variance"].values-df_down["variance"].values))/df_nominal["yield"].values
    #        ratio_err[np.isnan(ratio_err) | np.isinf(ratio_err) | (ratio_err==0.)] = 1.
    #        up_smooth = np.maximum(smooth(xvals, ratio, ratio_err), 0.)
    #        scale = df_up["yield"].sum() / (up_smooth*df_nominal["yield"]).sum()

    #        try:
    #            if pd.isnull(up_smooth).any():
    #                continue
    #            df.loc[(d, r, p, s+"Up", n), "yield"] = (df_nominal["yield"].values*scale*up_smooth)[np.isin(np.array(binning)[:-1], df_up["bin0_low"])]
    #        except ValueError:
    #            print(d, r, p, s+"Up", n)

    #        ratio = df_down["yield"].values/df_nominal["yield"].values
    #        ratio[np.isnan(ratio) | np.isinf(ratio)] = 1.
    #        ratio_err = np.sqrt(np.abs(df_down["variance"].values-df_nominal["variance"].values))/df_nominal["yield"].values
    #        ratio_err[np.isnan(ratio_err) | np.isinf(ratio_err) | (ratio_err==0.)] = 1.
    #        down_smooth = np.maximum(smooth(xvals, ratio, ratio_err), 0.)
    #        scale = df_down["yield"].sum() / (down_smooth*df_nominal["yield"]).sum()

    #        try:
    #            if pd.isnull(down_smooth).any():
    #                continue
    #            df.loc[(d, r, p, s+"Down", n), "yield"] = (df_nominal["yield"].values*scale*down_smooth)[np.isin(np.array(binning)[:-1], df_down["bin0_low"])]
    #        except ValueError:
    #            print(d, r, p, s+"Down", n)

    for (d, r, p, w, n), dfgrp in df.groupby(all_inds_no_bins):
        if r not in [k.GetName() for k in rfile.GetListOfKeys()]:
            rfile.mkdir(r)
        rfile.cd(r)

        bins = array('f', binning)
        name = p
        if w != "nominal":
            if w not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
                ROOT.gDirectory.mkdir(w)
            ROOT.gDirectory.cd(w)
        hist = ROOT.TH1D(name, name, len(bins)-1, bins)

        for idx in range(1, hist.GetNbinsX()+1):
            binlow = bins[idx-1]
            binupp = bins[idx]
            dfbin = dfgrp[(dfgrp[binlab+"_low"]==binlow) & (dfgrp[binlab+"_upp"]==binupp)]
            if not dfbin.empty:
                content = dfbin.iloc[0]["yield"]
                error = np.sqrt(dfbin.iloc[0]["variance"])
                if np.abs(content) <= 1e-7:
                    content = 1e-7
                    error = 1e-7
                hist.SetBinContent(idx, content)
                hist.SetBinError(idx, error)
        hist.Write()
        hist.Delete()
        rfile.cd()

    # Create all data_obs (just in case some are completely blinded)
    for r in df.index.get_level_values("region").unique():
        if r not in [k.GetName() for k in rfile.GetListOfKeys()]:
            rfile.mkdir(r)
        rfile.cd(r)

        if "data_obs" not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
            bins = array('f', binning)
            hist = ROOT.TH1D("data_obs", "data_obs", len(bins)-1, bins)
            hist.Write()
            hist.Delete()
        rfile.cd()

    rfile.Close()
    logger.info("Created {}".format("Zinv_METnoX-ShapeTemplates_{}.root".format(cfg["name"])))
    df = df.set_index([binlab+"_low", binlab+"_upp"], append=True)\
            .reorder_levels(all_inds)

    # df_obs
    df_obs = df[df.index.get_level_values("process").isin(["data_obs"])]["yield"]
    df_obs = df_obs.groupby([n for n in df_obs.index.names if "bin" not in n]).sum()

    # df_rate
    df_rate = df[df.index.get_level_values("weight").isin(["nominal"])]["yield"]
    df_rate = df_rate.groupby(df_rate.index.names).sum()
    df_rate[df_rate<0.] = np.nan
    df_rate[df_rate.groupby("region").apply(lambda x: x/x.sum())<0.0001] = np.nan
    df_rate = df_rate.dropna()
    df_rate = df_rate.groupby([n for n in df_rate.index.names if "bin" not in n]).sum()
    df_rate = df_rate.reset_index("process")
    df_rate["proc"] = df_rate["process"].map(cfg["processes"])
    df_rate = df_rate.set_index("process", append=True)\
            .reorder_levels([n for n in all_inds if "bin" not in n])

    # df_nuis
    df_nominal = df[df.index.get_level_values("weight").isin(["nominal"])]
    df_nominal = df_nominal.groupby(df_nominal.index.names).sum()
    df_nuis = df.pivot_table(values='yield', index=['region', 'process'],
                             columns=['weight'], aggfunc=np.sum)
    df_nominal = df_nominal.pivot_table(values='yield', index=['region', 'process'],
                                        columns=['weight'], aggfunc=np.sum)
    df_nuis = df_nuis.divide(df_nominal["nominal"], axis=0)

    df_nuis.loc[:,"lumiUp"] = 1.025
    df_nuis.loc[:,"lumiDown"] = 1/1.025
    #df_nuis.loc[df_nuis.index.get_level_values("process").isin(["wlnu"]),"lumiUp"] = 1.
    #df_nuis.loc[df_nuis.index.get_level_values("process").isin(["wlnu"]),"lumiDown"] = 1.
    df_nuis = df_nuis.fillna(1.)

    df_nuis = df_nuis[[
        ss for s in cfg["systematics"] for ss in (s+"Down", s+"Up")
        if ss in df_nuis.columns
    ]]

    proc_order = [
        (r, p)
        for r in cfg["regions"]
        for p in cfg["processes"].keys()
    ]
    bin_order = []
    for bin, proc in proc_order:
        if bin not in bin_order:
            bin_order.append(bin)
    df_obs = df_obs.reset_index(["dataset", "process", "weight", "name"], drop=True)
    df_rate = df_rate.reset_index(["dataset", "weight", "name"], drop=True)
    df_obs = df_obs.reindex(bin_order, fill_value=0)
    df_rate = df_rate.reindex(proc_order).dropna()
    df_nuis = df_nuis.loc[df_rate.index]

    create_shape_datacard(df_obs, df_rate, df_nuis, cfg["parameters"], "Zinv_METnoX-Shapes_{}.txt".format(cfg["name"]), cfg["name"])

def create_shape_datacard(df_obs, df_rate, df_nuis, params, filename, name):
    # IDX
    dc = tab([
        ["imax * number of channels"],
        ["jmax * number of backgrounds"],
        ["kmax * number of nuisance parameters"],
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # SHAPES
    df_obs = df_obs.reset_index()
    dc += tab([
        ["shapes", "*", "*", "Zinv_METnoX-ShapeTemplates_{}.root".format(name), "$CHANNEL/$PROCESS", "$CHANNEL/$SYSTEMATIC/$PROCESS"]
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # OBS
    dc += tab([
        ["bin"] + list(df_obs["region"]),
        ["observation"] + [-1]*df_obs.shape[0],
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # RATE
    df_rate = df_rate.reset_index()
    dc += tab([
        ["bin"] + list(df_rate["region"]),
        ["process"] + list(df_rate["process"]),
        ["process"] + map(int, list(df_rate["proc"])),
        ["rate"] + [-1]*df_rate.shape[0],
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # NUISANCES
    nuisances = []
    for c in df_nuis.columns:
        syst = c[:-2] if c.endswith("Up") else c[:-4] if c.endswith("Down") else c
        if syst not in nuisances and "nominal" not in syst:
            nuisances.append(syst)

    nuisance_block = []
    for nuis in nuisances:
        if nuis in ["lumi"]:
            nuisance_subblock = [nuis, "lnN"]
        else:
            nuisance_subblock = [nuis, "shape"]
        for up, down in zip(df_nuis[nuis+"Up"], df_nuis[nuis+"Down"]):
            if nuis in ["lumi"]:
                value = str(np.sqrt(up/down))
            else:
                value = 1

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

            nuisance_subblock.append(value)
        nuisance_block.append(nuisance_subblock)
    dc += tab(nuisance_block, [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # PARAMS
    if params is not None:
        dc += tab(params, [], tablefmt="plain")

    with open(filename, 'w') as f:
        f.write(dc)
    logger.info("Created {}".format(filename))

def create_counting_datacards(df, cfg):
    pass

def main():
    options = parse_args()

    with open(options.config, 'r') as f:
        config = yaml.load(f)
    config["input"] = options.input

    if options.systematics == 0:
        config["systematics"] = []

    df = open_df(config)
    for name, selection in config["selections"].items():
        df_s = apply_selection(df, selection)
        df_s = reformat(df_s, config)

        config["name"] = name
        if config["shape"]:
            create_shape_datacards(df_s, config)
        else:
            create_counting_datacards(df_s, config)

if __name__ == "__main__":
    main()

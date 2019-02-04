import argparse
import copy
import numpy as np
import oyaml as yaml
import pandas as pd
from array import array
from tabulate import tabulate as tab

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
    return parser.parse_args()

def open_df(cfg):
    path = cfg["input"]
    with open(path, 'r') as f:
        _, df = pickle.load(f)
    return df

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
        df_s = df_s.reset_index(["variable0", "bin0_low", "bin0_upp"], drop=True)
    if "bin1" in selection:
        df_s = df_s.reset_index(["variable1", "bin1_low", "bin1_upp"], drop=True)
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

    df = process_syst(df, syst="lhePdf",
                      how_up=lambda x: x.mean()+x.std(),
                      how_down=lambda x: x.mean()*x.mean()/(x.mean()+x.std()))
    df = process_syst(df, syst="lheScale",
                      how_up=lambda x: x.max(),
                      how_down=lambda x: x.min())

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
    varlab = binlab.replace("bin", "variable")
    binning = eval(binning)
    all_inds = df.index.names
    all_inds_no_bins = [i for i in all_inds if "bin" not in i]

    # open ROOT file
    rfile = ROOT.TFile.Open("Zinv_METnoX-ShapeTemplates_{}.root".format(cfg["name"]), 'RECREATE')

    df = df.reset_index([i for i in all_inds if "bin" in i])
    for (d, r, p, w, n, v), dfgrp in df.groupby(all_inds_no_bins):
        print(p)
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
                hist.SetBinContent(idx, content)
                hist.SetBinError(idx, error)
        hist.Write()
        rfile.cd()
    rfile.Close()
    logger.info("Created {}".format("Zinv_METnoX-ShapeTemplates_{}.root".format(cfg["name"])))
    exit()
    df = df.set_index([binlab+"_low", binlab+"_upp"], append=True)\
            .reorder_levels(all_inds)

    # df_obs
    df_obs = df[df.index.get_level_values("process").isin(["data_obs"])]["yield"]
    df_obs = df_obs.groupby([n for n in df_obs.index.names if "bin" not in n]).sum()

    # df_rate
    df_rate = df[df.index.get_level_values("weight").isin(["nominal"])]["yield"]
    df_rate = df_rate.groupby(df_rate.index.names).sum()
    df_rate = df_rate.reset_index("process")
    df_rate["proc"] = df_rate["process"].map(
        dict(zip(*zip(*enumerate(cfg["processes"], 0))[::-1]))
    )
    df_rate = df_rate.set_index("process", append=True)\
            .reorder_levels(all_inds)
    df_rate[df_rate["yield"]<0.] = np.nan
    df_rate[df_rate.groupby("region").apply(lambda x: x/x.sum())["yield"]<0.0001] = np.nan
    df_rate = df_rate.dropna()
    df_rate = df_rate.groupby([n for n in df_rate.index.names if "bin" not in n]).sum()

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
        c for c in df_nuis.columns
        if (c[:-2] if c.endswith("Up") else c[:-4]) in cfg["systematics"]
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
    df_obs = df_obs.reset_index(["dataset", "process", "weight", "name", varlab], drop=True)
    df_rate = df_rate.reset_index(["dataset", "weight", "name", varlab], drop=True)
    df_obs = df_obs.reindex(bin_order, fill_value=0)
    df_rate = df_rate.reindex(proc_order).dropna()
    df_nuis = df_nuis.loc[df_rate.index]

    create_shape_datacard(df_obs, df_rate, df_nuis, cfg["parameters"], "Zinv_METnoX-Shapes_{}.txt".format(cfg["name"]))

def create_shape_datacard(df_obs, df_rate, df_nuis, params, filename):
    # IDX
    dc = tab([
        ["imax * number of channels"],
        ["jmax * number of backgrounds"],
        ["kmax * number of nuisance parameters"],
    ], [], tablefmt="plain") + "\n" + "-"*80 + "\n"

    # SHAPES
    df_obs = df_obs.reset_index()
    dc += tab([
        ["shapes", "*", "*", "Zinv_METnoX-ShapeTemplates.root", "$CHANNEL/$PROCESS", "$CHANNEL/$SYSTEMATIC/$PROCESS"]
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
    nuisances = sorted(list(set((c[:-2] if c.endswith("Up") else c[:-4]) for c in df_nuis.columns)))
    nuisances = [n for n in nuisances if "nominal" not in n]

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

import argparse
from array import array
import copy
import numpy as np
from tabulate import tabulate as tab
import oyaml as yaml

import ROOT
ROOT.gROOT.SetBatch(True)

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
    parser.add_argument("config", type=str, help="Yaml config file")
    parser.add_argument("--binning", type=str, default="[200.]",
                        help="Binning to use")
    parser.add_argument("--shape", default=False, action='store_true',
                        help="Create shape datacards")

    return parser.parse_args()

def open_df(cfg):
    path = cfg["input"]
    with open(path, 'r') as f:
        _, df = pickle.load(f)
    df = df.reset_index("variable0", drop=True)
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

def reformat(df, cfg):
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
    binning = cfg["binning"]
    bins = [-np.infty]+list(binning)+[np.infty]
    df = rebin(df, bins)

    df = process_syst(df, syst="lhePdf", how_up=lambda x: x.mean()+x.std(), how_down=lambda x: x.mean()*x.mean()/(x.mean()+x.std()))
    df = process_syst(df, syst="lheScale", how_up=lambda x: x.max(), how_down=lambda x: x.min())

    if "dataset_region_process_conv" in cfg:
        levels = df.index.names
        df = df.reset_index(["dataset", "region", "process"])
        df["drp"] = df["dataset"]+"__"+df["region"]+"__"+df["process"]
        rp_map = {"__".join(k): "__".join(v) for k, v in cfg["dataset_region_process_conv"].items()}
        df["drp"] = df["drp"].map(rp_map, na_action='ignore')
        df = df[~df["drp"].isna()]
        df["dataset"] = df["drp"].apply(lambda x: x.split("__")[0])
        df["region"] = df["drp"].apply(lambda x: x.split("__")[1])
        df["process"] = df["drp"].apply(lambda x: x.split("__")[2])
        df = df.set_index(["dataset", "region", "process"], append=True)\
                .drop("drp", axis=1)\
                .reorder_levels(levels)

    if "apply" in cfg:
        for _, apply in cfg["apply"].items():
            levels = list(df.index.names)
            tdf = df.reset_index().set_index(apply["levels"])
            try:
                args = [
                    tdf.loc[tuple(arg),:].reset_index(drop=True)\
                    .set_index([l for l in levels if l not in apply["levels"]])
                    for arg in apply["args"]
                ]
            except KeyError:
                continue
            new_df = eval('lambda {}'.format(apply["formula"]))(*args)
            for idx, level in enumerate(apply["levels"]):
                new_df[level] = apply["new_levels"][idx]
            new_df = new_df.set_index(apply["levels"], append=True)\
                    .reorder_levels(levels)
            df = pd.concat([df, new_df], axis=0)

    # Rename regions
    def rename_level_values(df, level, name_map):
        levels = df.index.names
        df = df.reset_index(level)
        df[level] = df[level].map(name_map, na_action='ignore')
        df = df[~df[level].isna()]
        df = df.set_index(level, append=True)\
                .reorder_levels(levels)
        return df
    df = rename_level_values(df, "dataset", cfg["dataset_conv"])
    df = rename_level_values(df, "region", cfg["region_conv"])
    df = rename_level_values(df, "process", cfg["process_conv"])
    df = df.groupby(df.index.names).sum()

    levels = df.index.names
    df = df.reset_index(["process", "dataset"])
    df["drop"] = df.apply(lambda row: row["process"] in cfg["dataset_conv"].keys()
                          and row["process"] != row["dataset"], axis=1)
    df = df.set_index(["process", "dataset"], append=True)\
            .reorder_levels(levels)
    df = df[~df["drop"]].drop("drop", axis=1)

    # Select MET->(monojet, singlemu, doublemu) and
    # SingleElectron->(singleele, doubleele)
    df = df.reset_index(["process", "weight", "name", "bin0_low", "bin0_upp"])
    df = df.loc[cfg["dataset_regions"]].dropna()
    df = df.set_index(["process", "weight", "name", "bin0_low", "bin0_upp"], append=True)

    return df

def create_counting_datacards(df, cfg):
    allowed_datasets = cfg["dataset_conv"].values()
    for category, dfgroup in df.groupby(["bin0_low", "bin0_upp"]):
        if np.isinf(category[0]):
            continue

        # data obs
        df_obs = dfgroup[dfgroup.index.get_level_values("process").isin(allowed_datasets)]["yield"]
        df_obs = df_obs.reset_index([l for l in df.index.names if l != "region"], drop=True)
        dfgroup = dfgroup[~dfgroup.index.get_level_values("process").isin(allowed_datasets)]

        # mc rate
        df_rate = dfgroup[dfgroup.index.get_level_values("weight").isin(["nominal"])]["yield"]
        df_rate = df_rate.reset_index([l for l in df.index.names if l not in ["region", "process"]], drop=True)
        df_rate = df_rate.fillna(1e-10)
        df_rate[df_rate<0.] = np.nan
        df_rate[df_rate.groupby("region").apply(lambda x: x/x.sum())<0.001] = np.nan
        df_rate = df_rate.dropna()

        # nuisances
        df_nominal = dfgroup[dfgroup.index.get_level_values("weight").isin(["nominal"])]
        dfgroup = dfgroup[~dfgroup.index.get_level_values("weight").isin(["nominal"])]

        df_nuis = dfgroup.pivot_table(values='yield', index=['region', 'process'],
                                      columns=['weight'], aggfunc=np.sum)
        df_nominal = df_nominal.pivot_table(values='yield', index=['region', 'process'],
                                            columns=['weight'], aggfunc=np.sum)
        df_nuis = df_nuis.divide(df_nominal["nominal"], axis=0)

        if cfg["systematics"] and "lumi" in cfg["systematics"]:
            df_nuis.loc[:,"lumiUp"] = 1.025
            df_nuis.loc[:,"lumiDown"] = 1/1.025
            df_nuis.loc[df_nuis.index.get_level_values("process").isin(["wlnu"]),"lumiUp"] = 1.
            df_nuis.loc[df_nuis.index.get_level_values("process").isin(["wlnu"]),"lumiDown"] = 1.

        # Fix one-sided uncertainties (lack of stats in events which differ)
        nuisances = list(set(c[:-2] if c.endswith("Up") else c[:-4] for c in df_nuis.columns))
        nuisances = [n for n in nuisances if "lhe" not in n]
        for n in nuisances:
            df_temp = df_nuis.loc[:, [n+"Up", n+"Down"]]
            df_temp.loc[((df_temp-1).prod(axis=1)>0) & (np.abs(df_temp[n+"Up"]-1)>=np.abs(df_temp[n+"Down"]-1)), n+"Down"] = 1.
            df_temp.loc[((df_temp-1).prod(axis=1)>0) & (np.abs(df_temp[n+"Up"]-1)<np.abs(df_temp[n+"Down"]-1)), n+"Up"] = 1.
            df_nuis[[n+"Up", n+"Down"]] = df_temp

        # order
        regions = [r for d, r in cfg["dataset_regions"]]
        processes = cfg["processes"]
        proc_order = [
            (r, p)
            for r in regions
            for p in processes
        ]

        bin_order = []
        for bin, proc in proc_order:
            if bin not in bin_order:
                bin_order.append(bin)
        df_obs = df_obs.reindex(bin_order, fill_value=0)
        df_rate = df_rate.reindex(proc_order, fill_value=1e-10).dropna()
        df_rate = df_rate.reset_index("process")
        df_rate["proc"] = df_rate["process"].map(cfg["proc_id"])
        df_rate = df_rate.set_index("process", append=True)
        df_nuis = df_nuis.reindex(df_rate.index)

        df_nuis = df_nuis[[
            c for c in df_nuis.columns
            if (c[:-2] if c.endswith("Up") else c[:-4]) in cfg["systematics"]
        ]]

        low = int(category[0]) if not np.isinf(category[0]) else "Inf"
        high = int(category[1]) if not np.isinf(category[1]) else "Inf"
        filename = "Zinv_METnoX-{}To{}.txt".format(low, high)

        create_counting_datacard(df_obs, df_rate, df_nuis, cfg["parameters"], filename)

def create_counting_datacard(df_obs, df_rate, df_nuis, params, filename):
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
        ["observation"] + list(df_obs["yield"]), #map(int, list(df_obs["yield"])),
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
    nuisances = sorted(list(set(c[:-2] if c.endswith("Up") else c[:-4] for c in df_nuis.columns)))

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

    # PARAMS
    if params is not None:
        dc += tab([[k]+v for k, v in params.items()], [], tablefmt="plain")

    with open(filename, 'w') as f:
        f.write(dc)
    logger.info("Created {}".format(filename))

def create_shape_datacards(df, cfg):
    binning = cfg["binning"]
    all_columns = df.index.names
    columns_no_bins = [c for c in all_columns if "bin" not in c]

    # open ROOT file
    rfile = ROOT.TFile.Open("Zinv_METnoX-ShapeTemplates.root", 'RECREATE')

    # dataset, region, process, weight, variable
    df = df.reset_index(["bin0_low", "bin0_upp"])
    for (d, r, p, w, v), dfgrp in df.groupby(columns_no_bins):
        if r not in [k.GetName() for k in rfile.GetListOfKeys()]:
            rfile.mkdir(r)
        rfile.cd(r)

        bins = array('f', binning)
        name = p
        if w == "nominal":
            if d == p:
                name = "data_obs"
        else:
            if w not in [k.GetName() for k in ROOT.gDirectory.GetListOfKeys()]:
                ROOT.gDirectory.mkdir(w)
            ROOT.gDirectory.cd(w)
        hist = ROOT.TH1D(name, name, len(bins)-1, bins)

        for idx in range(1, hist.GetNbinsX()+1):
            binlow = bins[idx-1]
            binupp = bins[idx]
            dfbin = dfgrp[(dfgrp["bin0_low"]==binlow) & (dfgrp["bin0_upp"]==binupp)]
            if not dfbin.empty:
                content = dfbin.iloc[0]["yield"]
                error = np.sqrt(dfbin.iloc[0]["variance"])
                hist.SetBinContent(idx, content)
                hist.SetBinError(idx, error)
        hist.Write()
        rfile.cd()
    rfile.Close()
    logger.info("Created {}".format("Zinv_METnoX-ShapeTemplates.root"))

    # df_obs
    allowed_datasets = cfg["dataset_conv"].keys()
    df_obs = df[df.index.get_level_values("process").isin(allowed_datasets)]["yield"]
    df_obs = df_obs.groupby(df_obs.index.names).sum()

    # df_rate
    df_rate = df[df.index.get_level_values("weight").isin(["nominal"])]["yield"]
    df_rate = df_rate.groupby(df_rate.index.names).sum()
    df_rate = df_rate.reset_index("process")
    df_rate["proc"] = df_rate["process"].map(
        dict(zip(*zip(*enumerate(cfg["processes"], 0))[::-1]))
    )
    df_rate = df_rate.set_index("process", append=True)
    df_rate[df_rate["yield"]<0.] = np.nan
    df_rate[df_rate.groupby("region").apply(lambda x: x/x.sum())["yield"]<0.0001] = np.nan
    df_rate = df_rate.dropna()

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
    df_nuis.loc[df_nuis.index.get_level_values("process").isin(["wlnu"]),"lumiUp"] = 1.
    df_nuis.loc[df_nuis.index.get_level_values("process").isin(["wlnu"]),"lumiDown"] = 1.
    df_nuis = df_nuis.fillna(1.)

    df_nuis = df_nuis[[
        c for c in df_nuis.columns
        if (c[:-2] if c.endswith("Up") else c[:-4]) in cfg["systematics"]
    ]]

    regions = [r for d, r in cfg["dataset_regions"]]
    processes = cfg["processes"]
    proc_order = [
        (r, p)
        for r in regions
        for p in processes
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

    create_shape_datacard(df_obs, df_rate, df_nuis, cfg["parameters"], "Zinv_METnoX-Shapes.txt")

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
        dc += tab([[k]+v for k, v in params.items()], [], tablefmt="plain")

    with open(filename, 'w') as f:
        f.write(dc)
    logger.info("Created {}".format(filename))


def main():
    options = parse_args()

    with open(options.config, 'r') as f:
        config = yaml.load(f)
    config["input"] = options.input
    config["binning"] = eval(options.binning)
    config["shape"] = options.shape
    df = open_df(config)
    df = reformat(df, config)

    with open("trigger_df.pkl", 'w') as f:
        pickle.dump(df, f)

    if config["shape"]:
        create_shape_datacards(df, config)
    else:
        create_counting_datacards(df, config)

if __name__ == "__main__":
    main()

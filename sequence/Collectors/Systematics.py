import numpy as np
import os
import operator
import copy
try: import cPickle as pickle
except ImportError: import pickle

from utils.Histogramming import Histograms
from .Histogrammer import Config, HistReader, HistCollector
from drawing.dist_multicomp import dist_multicomp
from drawing.dist_facet import dist_facet

class SystematicsReader(HistReader):
    def begin(self, event):
        # Only run for variations defined in the event
        self.histograms.configs = [
            c for c in self.histograms.configs
            if c["weightname"] in event.variations\
            or not any(v in c["weightname"] for v in ["jes", "jer", "unclust"])
        ]
        if "lhe" not in event.optsysts:
            self.histograms.configs = [
                c for c in self.histograms.configs
                if not any(v in c["weightname"] for v in ["lhe"])
            ]
        else:
            self.histograms.configs = [
                c for c in self.histograms.configs
                if any(v in c["weightname"] for v in ["lhe"])
            ]
        super(SystematicsReader, self).begin(event)

class SystematicsCollector(HistCollector):
    def save(self, histograms):
        df = histograms.histograms

        # Remove variations from name and region
        levels = df.index.names
        df = df.reset_index(["name", "region", "weight"])
        df["name"] = df.apply(lambda row: row["name"].replace(row["weight"], ""), axis=1)
        df["region"] = df.apply(lambda row: row["region"].replace(row["weight"], ""), axis=1)
        df = df.set_index(["name", "region", "weight"], append=True).reorder_levels(levels)

        histograms.histograms = df
        histograms.save(self.outdir)

    def draw(self, histograms):
        datasets = ["MET", "SingleMuon", "SingleElectron"]
        df = histograms.histograms
        binning = histograms.binning

        def rename_level_values(df, level, name_map):
            levels = df.index.names
            df = df.reset_index(level)
            df[level] = df[level].map(name_map, na_action='ignore')
            df = df[~df[level].isna()]
            df = df.set_index(level, append=True).reorder_levels(levels)
            return df

        df = rename_level_values(df, "process", {
            "ZJetsToNuNu":      "znunu",      "DYJetsToMuMu":     "zmumu",
            "DYJetsToEE":       "zee",        "WJetsToENu":       "wlnu",
            "WJetsToMuNu":      "wlnu",       "WJetsToTauLNu":    "wlnu",
            "WJetsToTauHNu":    "wlnu",       "QCD":              "qcd",
            "TTJets":           "bkg",        "Diboson":          "bkg",
            "DYJetsToTauLTauL": "bkg",        "DYJetsToTauLTauH": "bkg",
            "DYJetsToTauHTauH": "bkg",        "EWKV2Jets":        "bkg",
            "SingleTop":        "bkg",        "G1Jet":            "bkg",
            "VGamma":           "bkg",        "MET":              "MET",
            "SingleMuon":       "SingleMuon", "SingleElectron":   "SingleElectron",
        })
        df = df.groupby(df.index.names).sum()

        all_columns = list(df.index.names)
        all_columns.insert(all_columns.index("weight"), "key")
        all_columns.remove("weight")
        all_columns.remove("variable0")
        columns_nobins = [c for c in all_columns if "bin" not in c]
        columns_nobins_nokey = [c for c in columns_nobins if c != "key"]
        columns_nobins_nokey_noproc = [c for c in columns_nobins_nokey if c != "process"]

        df = df.reset_index("variable0", drop=True)
        df = df.reset_index(["weight", "process"])
        df["key"] = df["weight"]
        df = df[~df["process"].isin(datasets)]
        df = df.drop("weight", axis=1)\
                .set_index("key", append=True)\
                .set_index("process", append=True)\
                .reorder_levels(all_columns)

        # Sort the key index
        df = df.reset_index(["key", "bin0_low"])
        sorter = list(df["key"].unique())
        sorter.remove("nominal")
        sorter.insert(0, "nominal")
        sorter_idx = dict(zip(sorter, range(len(sorter))))
        df["key_rank"] = df["key"].map(sorter_idx)
        df = df.sort_values(["bin0_low", "key_rank"], ascending=True)
        df = df.drop("key_rank", axis=1)\
                .set_index("key", append=True)\
                .set_index("bin0_low", append=True)\
                .reorder_levels(all_columns)\

        args = []
        #for categories, df_group in df.groupby(columns_nobins_nokey):
        #    # Create output directory structure
        #    path = os.path.join(self.outdir, "plots", *categories[:2])
        #    if not os.path.exists(path):
        #        os.makedirs(path)
        #    filepath = os.path.abspath(os.path.join(path, "__".join(categories[2:])))

        #    # Create args list for post-processing drawing
        #    cfg = copy.deepcopy(self.cfg)
        #    cfg.name = cfg.axis_label.get(categories[3], categories[3])
        #    bins = binning[categories[3]]
        #    with open(filepath+".pkl", 'w') as f:
        #        pickle.dump((df_group, bins, filepath, cfg), f)
        #    args.append((dist_multicomp, (df_group, bins, filepath, cfg)))

        # Apply rebinning ranks
        new_binning = [-np.inf,200,225,250,275,300,350,400,500,600,800,np.inf]
        df["rank"] = 0
        for bidx in range(len(new_binning)-1):
            df.loc[
                (df.index.get_level_values("bin0_low")>=new_binning[bidx])\
                & (df.index.get_level_values("bin0_upp")<=new_binning[bidx+1]),
                "rank"
            ] = bidx
        df = df.set_index("rank", append=True)
        df = df.groupby(columns_nobins+["rank"]).sum()
        df = df.reset_index("rank")

        df["bin0_low"] = np.nan
        df["bin0_upp"] = np.nan
        for bidx in range(len(new_binning)-1):
            df.loc[df["rank"]==bidx, "bin0_low"] = new_binning[bidx]
            df.loc[df["rank"]==bidx, "bin0_upp"] = new_binning[bidx+1]
        df = df.set_index(["bin0_low", "bin0_upp"], append=True)\
                .drop("rank", axis=1)

        for categories, df_group in df.groupby(columns_nobins_nokey_noproc):
            # Create output directory structure
            path = os.path.join(self.outdir, "plots", *categories[:2])
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.abspath(os.path.join(path, "__".join(categories[2:])))


            # mc stat
            df_unstack = df_group["yield"].unstack(level="key")
            df_mcstat = np.sqrt(df_group["variance"].unstack(level="key")["nominal"])
            df_unstack["mcstatUp"] = df_unstack["nominal"]+df_mcstat
            df_unstack["mcstatDown"] = df_unstack["nominal"]-df_mcstat

            # process order from total yield
            process_order = df_unstack["nominal"].groupby("process").sum()\
                    .sort_values(ascending=False).index.values

            # take ratio and remove nominal column
            df_unstack = df_unstack.divide(df_unstack["nominal"], axis="index")
            df_unstack = df_unstack[[c for c in df_unstack.columns if c != "nominal"]]

            # Create args list for post-processing drawing
            cfg = copy.deepcopy(self.cfg)
            cfg.process_order = process_order
            cfg.xlabel = cfg.axis_label.get(categories[2], categories[2])
            cfg.ylabel = "Relative uncertainty"
            cfg.name = categories[1]
            bins = binning[categories[2]]
            with open(filepath+".pkl", 'w') as f:
                pickle.dump((df_unstack, bins, filepath, cfg), f)
            args.append((dist_facet, (df_unstack, bins, filepath, cfg)))

        return args

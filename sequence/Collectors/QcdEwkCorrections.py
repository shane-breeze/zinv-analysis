import copy
import os
import numpy as np
import operator
try: import cPickle as pickle
except ImportError: import pickle

from drawing.dist_comp import dist_comp
from .Histogrammer import HistReader, HistCollector, Config

class QcdEwkCorrectionsReader(HistReader):
    def __init__(self, **kwargs):
        super(QcdEwkCorrectionsReader, self).__init__(**kwargs)
        self.split_samples = {}

class QcdEwkCorrectionsCollector(HistCollector):
    def draw(self, histograms):
        df = histograms.histograms
        binning = histograms.binning
        all_columns = list(df.index.names)
        all_columns.insert(all_columns.index("process"), "key")
        all_columns.remove("process")

        # Allowed processes are as shown below
        df = df.reset_index("process")
        df = df[df["process"].isin(["DYJetsToLL", "WJetsToLNu", "ZJetsToNuNu"])]
        df["key"] = df["process"]
        df = df.drop("process", axis=1)\
                .set_index("key", append=True)\
                .reorder_levels(all_columns)\
                .reset_index("weight", drop=True)

        columns_nokey = [c for c in all_columns if "key" not in c]
        columns_nokey_nobin = [c for c in columns_nokey if "bin0" not in c]
        columns_nokey_nobin_noname_noweight = [c for c in columns_nokey_nobin if c != "name" and c != "weight"]

        args = []
        for category, df_group in df.groupby(columns_nokey_nobin_noname_noweight):
            path = os.path.join(self.outdir, "plots", *category[:2])
            if not os.path.exists(path):
                os.makedirs(path)
            name = [i for i in set(df_group.index.get_level_values("name")) if "corrected" not in i].pop()
            filepath = os.path.abspath(os.path.join(path, name))
            cfg = copy.deepcopy(self.cfg)
            cfg.name = name
            bins = binning[name]
            with open(filepath+".pkl", 'w') as f:
                pickle.dump((df_group, bins, filepath, cfg), f)
            args.append((dist_comp, (df_group, bins, filepath, cfg)))

        return args

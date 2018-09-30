import copy
import os
import operator
import re

from drawing.dist_stitch import dist_stitch
from utils.Histogramming import Histograms

from Histogrammer import Config, HistReader, HistCollector

class GenStitchingReader(HistReader):
    def begin(self, event):
        parent = event.config.dataset.name.split("_ext")[0]
        self.parents = [parent]
        self.histograms.begin(event, self.parents, {})

processes = [
    "DYJetsToLL",
    "WJetsToLNu",
    "ZJetsToNuNu",
    "G1Jet",
    "QCD",
]

class GenStitchingCollector(HistCollector):
    def draw(self, histograms):
        df = histograms.histograms

        all_columns = [i for i in df.index.names]
        all_columns.insert(all_columns.index("process")+1, "parent")
        columns_no_process = [c for c in all_columns if "process" not in c]
        columns_no_process_no_bin = [c for c in columns_no_process if "bin0" not in c]

        df = df.reset_index("process")
        df["parent"] = df["process"].apply(lambda p: next((proc for proc in processes if proc in p), "unknown"))
        df = df.set_index(["process", "parent"], append=True)\
                .reorder_levels(all_columns)

        args = []
        for category, df_group in df.groupby(columns_no_process_no_bin):
            path = os.path.join(self.outdir, "plots", *category[:2])
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.abspath(os.path.join(path, "__".join([category[2],category[4]])))
            cfg = copy.deepcopy(self.cfg)
            cfg.name = category[4]
            args.append((dist_stitch, (df_group, filepath, cfg)))

        return args

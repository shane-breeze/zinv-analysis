import os
import operator
import copy

from utils.Histogramming import Histograms
from Histogrammer import Config, HistReader, HistCollector
from drawing.dist_multicomp import dist_multicomp

SystematicsReader = HistReader

class SystematicsCollector(HistCollector):
    def draw(self, histograms):
        datasets = ["MET", "SingleMuon", "SingleElectron"]

        df = histograms.histograms
        all_columns = list(df.index.names)
        all_columns.insert(all_columns.index("weight"), "key")
        all_columns.remove("weight")
        all_columns.remove("variable0")
        columns_nobins = [c for c in all_columns if "bin" not in c]
        columns_nobins_nokey = [c for c in columns_nobins if c != "key"]

        # Remove variations from name and region
        df = df.reset_index("variable0", drop=True)
        df = df.reset_index(["name", "region", "weight", "process"])
        df["name"] = df.apply(lambda row: row["name"].replace(row["weight"], ""), axis=1)
        df["region"] = df.apply(lambda row: row["region"].replace(row["weight"], ""), axis=1)
        df["key"] = df["weight"]
        df = df[~df["process"].isin(datasets)]
        df = df.drop("weight", axis=1)\
                .set_index("name", append=True)\
                .set_index("region", append=True)\
                .set_index("key", append=True)\
                .set_index("process", append=True)\
                .reorder_levels(all_columns)

        args = []
        for categories, df_group in df.groupby(columns_nobins_nokey):
            # Create output directory structure
            path = os.path.join(self.outdir, "plots", *categories[:2])
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.abspath(os.path.join(path, "__".join(categories[2:])))

            # Create args list for post-processing drawing
            cfg = copy.deepcopy(self.cfg)
            cfg.name = cfg.axis_label.get(categories[3], categories[3])
            args.append((dist_multicomp, (df_group, filepath, cfg)))

        return args

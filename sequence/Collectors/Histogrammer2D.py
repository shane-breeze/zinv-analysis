import os
import copy
import cPickle as pickle

from Histogrammer import Config, HistReader, HistCollector
from drawing.dist_2d import dist_2d

Hist2DReader = HistReader

class Hist2DCollector(HistCollector):
    def draw(self, histograms):
        datasets = ["MET", "SingleMuon", "SingleElectron"]

        df = histograms.histograms
        binning = histograms.binning

        all_columns = list(df.index.names)
        columns_nobins = [c for c in all_columns if "bin" not in c]

        args = []
        for categories, df_group in df.groupby(columns_nobins):
            # Create output directory structure
            path = os.path.join(self.outdir, "plots", *categories[:4])
            if "_remove_" in path:
                path = path.replace("_remove_", "/remove_")
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.abspath(os.path.join(path, categories[4]))

            xlab, ylab = categories[4].split("__")
            cfg = copy.deepcopy(self.cfg)
            cfg.xlabel = cfg.axis_label.get(xlab, xlab)
            cfg.ylabel = cfg.axis_label.get(ylab, ylab)
            bins = binning[categories[4]]

            # Create args list for post-processing drawing
            with open(filepath+".pkl", 'w') as f:
                pickle.dump((df_group, bins, filepath, cfg), f)
            args.append((dist_2d, (df_group, bins, filepath, cfg)))
        return args

import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

from Histogrammer import HistReader, HistCollector
from zinv.drawing.dist_multicomp_v2 import dist_multicomp_v2

class GStarCorrectionReader(HistReader):
    def begin(self, event):
        self.histograms.begin(event, [event.config.dataset.parent], {})

class GStarCorrectionCollector(HistCollector):
    def draw(self, histograms):
        df = histograms.histograms
        binning = histograms.binning
        all_indices = list(df.index.names)

        # draw nominal only
        df = df[df.index.get_level_values("weight").isin(["nominal"])]
        df.index.names = [n if n != "process" else "key" for n in df.index.names]
        all_indices[all_indices.index("process")] = "key"

        df_z = df[df.index.get_level_values("key").isin(["ZJetsToLL"])].reset_index("key", drop=True)
        df_g = df[df.index.get_level_values("key").isin(["GStarJetsToLL"])].reset_index("key", drop=True)
        df_z, df_g = df_z.align(df_g, fill_value=0.)
        df_zg = (df_z + df_g)
        df_zg["key"] = "Z+GStarJetsToLL"
        df_zg = df_zg.set_index("key", append=True).reorder_levels(all_indices)
        df = df.append(df_zg)

        args = []
        for (d, r, w, n), df_group in df.groupby(["dataset", "region", "weight", "name"]):
            path = os.path.join(self.outdir, "plots", d, r)
            if not os.path.exists(path):
                os.makedirs(path)

            filepath = os.path.abspath(os.path.join(path, n))
            if w != "": filepath += "_" + w

            bins = binning[n][0]
            with open(filepath+".pkl", 'w') as f:
                pickle.dump((df_group, bins, filepath, self.cfg), f)
            args.append((dist_multicomp_v2, (df_group, bins, filepath, self.cfg)))
        return args

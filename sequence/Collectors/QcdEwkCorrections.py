import copy
import os
import numpy as np
import operator

from drawing.dist_comp import dist_comp
from Histogrammer import HistReader, HistCollector, Config

class QcdEwkCorrectionsReader(HistReader):
    def __init__(self, **kwargs):
        super(QcdEwkCorrectionsReader, self).__init__(**kwargs)
        self.split_samples = {}

class QcdEwkCorrectionsCollector(HistCollector):
    def draw(self, histograms):
        df = histograms.histograms

        all_columns = [i for i in df.index.names]
        columns_noprocess = [c for c in all_columns if "process" not in c]
        columns_noprocess_nobin = [c for c in columns_noprocess if "bin0" not in c]
        columns_noprocess_nobin_noname_noweight = [c for c in columns_noprocess_nobin if c != "name" and c != "weight"]

        args = []
        for category, df_group in df.groupby(columns_noprocess_nobin_noname_noweight):
            path = os.path.join(self.outdir, "plots", *category[:2])
            if not os.path.exists(path):
                os.makedirs(path)
            name = [i for i in set(df_group.index.get_level_values("name")) if "corrected" not in i].pop()
            filepath = os.path.abspath(os.path.join(path, name))
            cfg = copy.deepcopy(self.cfg)
            cfg.name = name
            args.append((dist_comp, (df_group, filepath, cfg)))

        return args

import os
import operator
import pandas as pd

from drawing.dist_ratio import dist_ratio
from utils.Histogramming import Histograms

# Take the cfg module and drop unpicklables
class Config(object):
    def __init__(self, sample_names=[], sample_colours=[], axis_label=[], log=False):
        self.sample_names = sample_names
        self.sample_colours = sample_colours
        self.axis_label = axis_label
        self.log = log

class HistReader(object):
    split_samples = {
        "DYJetsToLL": {
            "DYJetsToEE": ["ev: ev.LeptonIsElectron"],
            "DYJetsToMuMu": ["ev: ev.LeptonIsMuon"],
            "DYJetsToTauTau": ["ev: ev.LeptonIsTau"],
        },
        "WJetsToLNu": {
            "WJetsToENu": ["ev: ev.LeptonIsElectron"],
            "WJetsToMuNu": ["ev: ev.LeptonIsMuon"],
            "WJetsToTauNu": ["ev: ev.LeptonIsTau"],
        },
    }
    def __init__(self, **kwargs):
        cfg = kwargs.pop("cfg")
        self.cfg = Config(
            sample_names = cfg.sample_names,
            sample_colours = cfg.sample_colours,
            axis_label = cfg.axis_label,
            log = True,
        )
        self.__dict__.update(kwargs)
        self.histograms = self.create_histograms(cfg)

    def create_histograms(self, cfg):
        configs = []
        for cfg in cfg.histogrammer_cfgs:
            # expand categories
            for dataset, cutflow in cfg["categories"]:
                cutflow_restriction = "ev: ev.Cutflow_{}".format(cutflow)
                selection = [cutflow_restriction]
                if "additional_selection" in cfg:
                    selection.extend(cfg["additional_selection"])
                for weightname, weight in cfg["weights"]:
                    weight = weight.format(dataset=dataset)
                    identifier = (dataset, cutflow, None, cfg["name"], weightname)

                    configs.append({
                        "name": cfg["name"],
                        "dataset": dataset,
                        "region": cutflow,
                        "weight": weight,
                        "selection": selection,
                        "variables": cfg["variables"],
                        "bins": cfg["bins"],
                    })

        # Histograms collection
        histograms = Histograms()
        histograms.extend(configs)
        return histograms

    def begin(self, event):
        parent = event.config.dataset.parent
        self.parents = self.split_samples[parent].keys() \
                       if parent in self.split_samples \
                       else [parent]
        selection = self.split_samples[parent] \
                    if parent in self.split_samples \
                    else {}
        self.histograms.begin(event, self.parents, selection)

    def end(self):
        self.histograms.end()

    def event(self, event):
        self.histograms.event(event)

    def merge(self, other):
        self.histograms.merge(other.histograms)

class HistCollector(object):
    def __init__(self, **kwargs):
        # drop unpicklables
        cfg = kwargs.pop("cfg")
        self.cfg = Config(
            sample_names = cfg.sample_names,
            sample_colours = cfg.sample_colours,
            axis_label = cfg.axis_label,
            log = True,
        )
        self.__dict__.update(kwargs)

    def collect(self, dataset_readers_list):
        self.outdir = os.path.join(self.outdir, self.name)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        histograms = None
        for dataset, readers in dataset_readers_list:
            # Get histograms
            if histograms is None:
                histograms = readers[0].histograms
            else:
                histograms.merge(readers[0].histograms)

        histograms.save(self.outdir)
        if self.plot:
            try:
                return self.draw(histograms)
            except Exception as e:
                print(e)
        return []

    def draw(self, histograms):
        datasets = ["MET", "SingleMuon", "SingleElectron"]

        df = histograms.histograms
        all_columns = list(df.index.names)
        columns_noproc = [c for c in all_columns if c != "process"]
        columns_noproc_nobins = [c for c in columns_noproc if "bin" not in c]

        # Group into (dataset, region, weight, names, variable0)
        #df.groupby(columns_noproc_nobins).apply(dist_ratio, outdir, self.cfg)

        args = []
        for categories, df_group in df.groupby(columns_noproc_nobins):
            # Create output directory structure
            path = os.path.join(self.outdir, "plots", *categories[:2])
            if "_remove_" in path:
                path = path.replace("_remove_", "/remove_")
            if not os.path.exists(path):
                os.makedirs(path)
            filepath = os.path.abspath(os.path.join(path, categories[3]))

            # Create args list for post-processing drawing
            args.append((dist_ratio, (df_group, filepath, self.cfg)))
        return args

    def reload(self, outdir):
        self.outdir = os.path.join(outdir, self.name)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        histograms = Histograms()
        histograms.reload(os.path.join(outdir, self.name))
        return self.draw(histograms)

import os
import operator
import yaml
import numpy as np
import pandas as pd
import cPickle as pickle
pi = np.pi + 1e-10

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
    def __init__(self, **kwargs):
        with open(kwargs.pop("cfg"), 'r') as f:
            cfg = yaml.load(f)
        with open(kwargs.pop("drawing_cfg"), 'r') as f:
            drawing_cfg = yaml.load(f)
        self.cfg = Config(
            sample_names = drawing_cfg["sample_names"],
            sample_colours = drawing_cfg["sample_colours"],
            axis_label = drawing_cfg["axis_label"],
            log = True,
        )
        self.__dict__.update(kwargs)
        self.categories = self.create_categories(cfg["categories"])
        self.histograms = self.create_histograms(cfg)

    def create_categories(self, cfg):
        all_cat = []
        for _, catlist in cfg.items():
            all_cat.extend(catlist)
        cfg["all"] = all_cat
        return cfg

    def create_histograms(self, cfg):
        configs = []
        for name, config in cfg["configs"].items():
            # expand categories
            cats = []
            for c in config["categories"]:
                cats.extend(self.categories[c])
            for dataset, cutflow in cats:
                cutflow_restriction = "ev: ev.Cutflow_{}(ev)".format(cutflow)
                selection = [cutflow_restriction]
                if "additional_selection" in config:
                    selection.extend(config["additional_selection"])
                for weightname, nsig, source, weight in config["weights"]:
                    configs.append({
                        "name": name,
                        "dataset": dataset,
                        "region": cutflow,
                        "weight": weight,
                        "nsig": nsig,
                        "source": source,
                        "weightname": weightname,
                        "selection": selection,
                        "variables": config["variables"],
                        "bins": [
                            [-np.inf]+list(np.linspace(*bs))+[np.inf]
                            for bs in config["bins"]
                        ],
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
        with open(kwargs.pop("cfg"), 'r') as f:
            cfg = yaml.load(f)
        with open(kwargs.pop("drawing_cfg"), 'r') as f:
            drawing_cfg = yaml.load(f)
        self.cfg = Config(
            sample_names = drawing_cfg["sample_names"],
            sample_colours = drawing_cfg["sample_colours"],
            axis_label = drawing_cfg["axis_label"],
            log = True,
        )
        self.__dict__.update(kwargs)

    def collect(self, dataset_readers_list):
        self.outdir = os.path.join(self.outdir, self.name)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        histograms = None
        for dataset, readers in dataset_readers_list:
            if len(readers)==0:
                continue
            # Get histograms
            if histograms is None:
                histograms = readers[0].histograms
            else:
                histograms.merge(readers[0].histograms)
        self.save(histograms)
        if self.plot:
            try:
                return self.draw(histograms)
            except Exception as e:
                print(e)
        return []

    def save(self, histograms):
        histograms.save(self.outdir)

    def draw(self, histograms):
        datasets = ["MET", "SingleMuon", "SingleElectron"]

        df = histograms.histograms
        binning = histograms.binning
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

            if categories[2] != "":
                filepath += "_" + categories[2]

            # Create args list for post-processing drawing
            bins = binning[categories[3]][0]
            with open(filepath+".pkl", 'w') as f:
                pickle.dump((df_group, bins, filepath, self.cfg), f)
            args.append((dist_ratio, (df_group, bins, filepath, self.cfg)))
        return args

    def reload(self, outdir):
        self.outdir = os.path.join(outdir, self.name)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        histograms = Histograms()
        histograms.reload(os.path.join(outdir, self.name))
        return self.draw(histograms)

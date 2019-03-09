import copy
import numpy as np
import numba as nb
np.random.seed(123456)
import pandas as pd
import os
import cPickle as pickle
from Lambda import Lambda
import itertools
import operator

from .NumbaFuncs import get_bin_indices

@nb.njit
def numba_histogram(event_attrs, mins, maxs, weights):
    hist = np.zeros_like(mins[0], dtype=np.float32)
    indices = get_bin_indices(event_attrs, mins, maxs, 1)[:,0]

    for iev, idx in enumerate(indices):
        if idx >= 0:
            w = weights[iev]
            if w != 0.:
                hist[int(idx)] += weights[iev]

    return hist

class Histograms(object):
    def __init__(self):
        self.histograms = None
        self.configs = []
        self.lambda_functions = {}
        self.binning = {}

    def extend(self, configs):
        self.configs.extend(configs)

        # update binning
        for config in self.configs:
            name = config["name"]
            if isinstance(name, list):
                name = "__".join(name)
            self.binning[name] = config["bins"]

    def begin(self, event, parents, selection):
        self.isdata = event.config.dataset.isdata

        full_configs = []
        for config in self.configs:
            for v in config["variables"]:
                if v not in self.lambda_functions:
                    self.lambda_functions[v] = Lambda(v)

            # deal with multiple processes per dataset
            for parent in parents:
                full_selection = config["selection"][:]
                if parent in selection:
                    full_selection += selection[parent]
                if self.isdata:
                    full_selection = ["ev: ev.Is{}Triggered(ev)".format(config["dataset"])] + full_selection

                new_config = copy.deepcopy(config)
                new_config["process"] = parent
                if self.isdata:
                    new_config["nsig"] = 0.
                    new_config["source"] = ''
                function = reduce(operator.add, [Lambda(s) for s in full_selection])
                self.lambda_functions[function.function] = function
                new_config["selection"] = function.function

                function = Lambda(config["weight"])
                self.lambda_functions[function.function] = function
                new_config["weight"] = function.function
                full_configs.append(new_config)

        self.full_configs = sorted(
            full_configs, key=operator.itemgetter(
                "weightname", "dataset", "region", "process", "name",
            ),
        )
        return self

    def end(self):
        self.clear_empties()
        self.lambda_functions = None
        return self

    def clear_empties(self):
        df = self.histograms
        columns = [c for c in df.index.names if "bin" not in c]
        self.histograms = df.loc[df.groupby(columns)["count"].transform(func=np.sum)>0,:]

    def event(self, event):
        dfs = []
        for config in self.full_configs:
            weightname = config["weightname"].lower()
            if self.isdata and ("up" in weightname or "down" in weightname):
                continue

            event.nsig = config["nsig"]
            event.source = config["source"]
            df = self.generate_dataframe(event, config)
            dfs.append(df)

        if len(dfs) == 0:
            return self

        columns = [c for c in dfs[0].columns if c not in ["count", "yield", "variance"]]
        histograms = pd.concat(dfs).groupby(columns).sum()
        if self.histograms is None:
            self.histograms = histograms
        else:
            self.histograms = pd.concat([self.histograms, histograms])\
                    .groupby(columns)\
                    .sum()
        return self

    def generate_dataframe(self, event, config):
        selection = self.lambda_functions[config["selection"]](event)
        if self.isdata:
            weight = selection.astype(float)
        else:
            weight = self.lambda_functions[config["weight"]](event)*selection

        variables = []
        for idx, v in enumerate(config["variables"]):
            try:
                variables.append(self.lambda_functions[v](event).astype(np.float32))
            except AttributeError:
                temp = np.empty(ev.size, dtype=float)
                temp[:] = np.nan
                variables.append(temp.astype(np.float32))

        weights1 = weight
        weights2 = weight**2

        bins = [np.array(b) for b in config["bins"]]
        mins = [b[:-1] for b in bins]
        maxs = [b[1:] for b in bins]

        mins = np.meshgrid(*mins)
        maxs = np.meshgrid(*maxs)
        hist_bins = bins
        hist_counts = numba_histogram(variables, mins, maxs, np.ones_like(weights1))
        hist_yields = numba_histogram(variables, mins, maxs, weights1)
        hist_variance = numba_histogram(variables, mins, maxs, weights2)

        bin_names = [["bin{}_low".format(idx), "bin{}_upp".format(idx)]
                     for idx in reversed(list(range(len(hist_bins))))]
        bin_names = reduce(lambda x,y: x+y, bin_names)
        bin_names = ["bin0_low", "bin0_upp"]
        data = {
            "count": hist_counts,
            "yield": hist_yields,
            "variance": hist_variance
        }
        data.update({"bin{}_low".format(idx): mins[idx] for idx in range(len(mins))})
        data.update({"bin{}_upp".format(idx): mins[idx] for idx in range(len(maxs))})
        df = pd.DataFrame(data, columns=bin_names+["count", "yield", "variance"])

        df["dataset"] = config["dataset"]
        df["region"] = config["region"]
        df["process"] = config["process"]
        df["weight"] = config["weightname"]
        df["name"] = config["name"] if not isinstance(config["name"], list) else "__".join(config["name"])
        for idx, variable in reversed(list(enumerate(config["variables"]))):
            df["variable{}".format(idx)] = variable
        columns = [c for c in df.columns if c not in ["count", "yield", "variance"] and "bin" not in c]
        columns += bin_names + ["count", "yield", "variance"]
        df = df[columns]
        return self.make_sparse_df(df)

    def make_sparse_df(self, df):
        return df.loc[df["count"]!=0]

    def make_dense_df(self, df):
        pass

    def merge(self, other):
        if self.histograms.shape[0] == 0:
            return other
        elif other.histograms.shape[0] == 0:
            return self
        columns = self.histograms.index.names
        self.histograms = pd.concat([self.histograms, other.histograms])\
                .groupby(columns)\
                .sum()
        return self

    def save(self, outdir):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        path = os.path.join(outdir, "results.pkl")
        with open(path, 'w') as f:
            pickle.dump((self.binning, self.histograms), f, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def reload(self, outdir):
        path = os.path.join(outdir, "results.pkl")
        with open(path, 'r') as f:
            self.binning, self.histograms = pickle.load(f)
        return self

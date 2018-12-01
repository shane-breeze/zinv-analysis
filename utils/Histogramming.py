import copy
import numpy as np
np.random.seed(123456)
import pandas as pd
import os
import cPickle as pickle
from Lambda import Lambda
import itertools

class Histograms(object):
    def __init__(self):
        self.histograms = None
        self.configs = []
        self.string_to_func = {}
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

        funcs = []
        full_configs = []
        for config in self.configs:
            # deal with multiple processes per dataset
            for parent in parents:
                full_selection = config["selection"][:]
                if parent in selection:
                    full_selection += selection[parent]

                funcs.extend([
                    f for f in config["variables"]+full_selection+[config["weight"]]
                ])

                new_config = copy.deepcopy(config)
                new_config["process"] = parent
                new_config["selection"] = full_selection
                full_configs.append(new_config)

        self.string_to_func = {func: Lambda(func) for func in set(funcs)}
        self.full_configs = full_configs
        return self

    def end(self):
        self.clear_empties()
        self.string_to_func = {}
        return self

    def clear_empties(self):
        df = self.histograms
        columns = [c for c in df.index.names if "bin" not in c]
        self.histograms = df.loc[df.groupby(columns)["count"].transform(func=np.sum)>0,:]

    def event(self, event):
        dfs = []
        for config in self.full_configs:
            weight = config["weight"].lower()
            if self.isdata and ("up" in weight or "down" in weight or "lhepdf" in weight or "lhescale" in weight):
                continue

            df = self.generate_dataframe(event, config)
            dfs.append(df)

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
        selection = reduce(lambda x,y: x & y, [
            self.string_to_func[s](event)
            for s in config["selection"]
        ]) if len(config["selection"])>0 else np.array([True]*event.size)

        weight = self.string_to_func[config["weight"]](event)[selection]

        variables = []
        for idx, v in enumerate(config["variables"]):
            try:
                variables.append(self.string_to_func[v](event)[selection])
            except AttributeError:
                temp = np.empty((int(selection.sum())))
                temp[:] = np.nan
                variables.append(temp)

        weights1 = weight
        weights2 = weight**2

        variables = np.transpose(np.array(variables))
        bins = [np.array(b) for b in config["bins"]]

        hist_counts, hist_bins = np.histogramdd(variables, bins=bins)
        hist_yields = np.histogramdd(variables, bins=bins, weights=weights1)[0]
        hist_variance = np.histogramdd(variables, bins=bins, weights=weights2)[0]

        data = self.create_onedim_hists(
            hist_bins, hist_counts, hist_yields, hist_variance,
        )
        bin_names = [["bin{}_low".format(idx), "bin{}_upp".format(idx)]
                     for idx in reversed(list(range(len(hist_bins))))]
        bin_names = reduce(lambda x,y: x+y, bin_names)
        df = pd.DataFrame(
            data,
            columns = bin_names+["count", "yield", "variance"],
        )

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

    def create_onedim_hists(self, bins, counts, yields, variance):
        counts_1d = counts.T.ravel()
        counts_1d = counts_1d.reshape((counts_1d.shape[0],1))
        yields_1d = yields.T.ravel()
        yields_1d = yields_1d.reshape((yields_1d.shape[0],1))
        variance_1d = variance.T.ravel()
        variance_1d = variance_1d.reshape((variance_1d.shape[0],1))

        tbins = bins[::-1]
        bin_idxs = itertools.product(*[range(len(bin)-1) for bin in tbins])
        bins_1d = np.array([
            reduce(lambda x,y: x+y, [
                [tbins[dim][sub_bin_idx], tbins[dim][sub_bin_idx+1]]
                for dim, sub_bin_idx in enumerate(bin_idx)
            ])
            for bin_idx in bin_idxs
        ])
        return np.hstack([bins_1d, counts_1d, yields_1d, variance_1d])

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

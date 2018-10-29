from utils.Lambda import Lambda
from utils.NumbaFuncs import get_bin_indices
import numpy as np
import pandas as pd

class WeightObjects(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # Add dataframes to correctors
        for corrector in self.correctors:
            dfs = []
            for w, path in corrector["weighted_paths"]:
                ndims = len(corrector["binning_variables"])
                file_form = [("bin{}_low".format(idx), "bin{}_upp".format(idx))
                             for idx in range(ndims)]
                file_form = [bin_label
                             for bin_pair in file_form
                             for bin_label in bin_pair] + ["corr", "unc_up", "unc_down"]
                df = pd.DataFrame(read_file(path, file_form))

                for idx in range(ndims):
                    bin_low = "bin{}_low".format(idx)
                    bin_upp = "bin{}_upp".format(idx)
                    df.loc[df[bin_low]==df[bin_low].min(), bin_low] = -np.inf
                    df.loc[df[bin_upp]==df[bin_upp].max(), bin_upp] =  np.inf

                df["weight"] = w
                dfs.append(df)
            corrector["df"] = pd.concat(dfs)

    def begin(self, event):
        funcs = [corrector["binning_variables"] for corrector in self.correctors]
        self.string_to_func = {func: Lambda(func) for func in [f for fs in funcs for f in fs]}

    def end(self):
        self.string_to_func = {}

    def event(self, event):
        event_sfs = {}
        for corrector in self.correctors:
            # Get required objects
            df = corrector["df"]
            nweight = df["weight"].unique().shape[0]

            name = corrector["name"]
            collection = getattr(event, corrector["collection"])
            vars = corrector["binning_variables"]
            any_pass = corrector["any_pass"] if "any_pass" in corrector else False
            add_syst = corrector["add_syst"] if "add_syst" in corrector else 0.
            event_vars = [self.string_to_func[v](collection) for v in vars]

            # Select bin from reference table
            indices = None
            scale = 1
            for idx in range(len(vars)):
                bin_low = np.unique(df["bin{}_low".format(idx)].values)
                bin_upp = np.unique(df["bin{}_upp".format(idx)].values)

                ind = get_bin_indices(event_vars[idx], bin_low, bin_upp)
                if indices is None:
                    indices = ind
                else:
                    indices += scale*ind
                scale *= bin_low.shape[0]
            dfw = df.loc[indices]

            # Add event and object index to event correction table
            evidx = np.zeros_like(collection.pt.content, dtype=int)
            objidx = np.zeros_like(collection.pt.content,dtype=int)
            for idx, (start, stop) in enumerate(zip(collection.starts, collection.stops)):
                evidx[start:stop] = idx
                for subidx in range(start, stop):
                    objidx[subidx] = subidx-start
            evidx = np.vstack([evidx]*nweight).T.ravel()
            objidx = np.vstack([objidx]*nweight).T.ravel()
            dfw["evidx"] = evidx
            dfw["objidx"] = objidx

            # Evaluate relevant weighted mean variables
            dfw = dfw.set_index(["evidx", "objidx"])
            dfw["w_corr"] = dfw.eval("weight*corr")
            dfw["w_up"] = dfw.eval("(weight*unc_up)**2")
            dfw["w_down"] = dfw.eval("(weight*unc_down)**2")
            dfw = dfw.loc[:,["weight", "w_corr", "w_up", "w_down"]]
            dfw = dfw.groupby(["evidx", "objidx"]).sum()
            dfw["w_up"] = np.sqrt(dfw["w_up"])
            dfw["w_down"] = np.sqrt(dfw["w_down"])
            dfw = dfw.divide(dfw["weight"], axis=0)

            # Calculate event uncertainties
            dfw["add_syst"] = add_syst
            dfsf = pd.concat([
                np.sqrt(dfw.eval("w_up**2 + (w_corr*add_syst)**2").groupby("evidx").sum()),
                np.sqrt(dfw.eval("w_down**2 + (w_corr*add_syst)**2").groupby("evidx").sum()),
            ], axis=1)
            dfsf.columns = ["sf_up", "sf_down"]

            # Calculate event correction - include any_pass option for trigger
            # type SF
            if any_pass:
                dfw["w_corr"] = 1-dfw["w_corr"]
            dfsf["sf"] = dfw["w_corr"].groupby("evidx").prod()
            if any_pass:
                dfsf["sf"] = 1-dfsf["sf"]

            # variations
            sf_up = np.zeros(event.size)
            sf_down = np.zeros(event.size)
            sf_up[dfsf.index.get_level_values("evidx")] = dfsf["sf_up"]/dfsf["sf"]
            sf_down[dfsf.index.get_level_values("evidx")] = dfsf["sf_down"]/dfsf["sf"]

            setattr(event, "Weight_{}Up".format(name), 1+sf_up)
            setattr(event, "Weight_{}Down".format(name), 1-sf_down)

            # Central SF
            sf = np.ones(event.size)
            sf[dfsf.index.get_level_values("evidx")] = dfsf["sf"]
            event_sfs[name] = sf

        event_sfs = pd.DataFrame(event_sfs)
        for dataset, eval_key in self.dataset_applicators.items():
            setattr(event, "Weight_{}".format(dataset),
                    getattr(event, "Weight_{}".format(dataset))*event_sfs.eval(eval_key).values)

def read_file(path, form):
    with open(path, 'r') as f:
        try:
            lines = [map(float, l.strip().split())
                     for l in f.read().splitlines()
                     if l.strip()[0]!="#"]
        except ValueError as e:
            raise ValueError("{} for {}".format(e, path))
    return {k: v for k, v in zip(form, zip(*lines))}

import numpy as np
import numba as nb
import pandas as pd
import awkward as awk
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.Lambda import Lambda
from zinv.utils.NumbaFuncs import weight_numba, get_bin_indices

def evaluate_object_weights(
    ev, source, nsig, df, bins_vars, add_syst, name,
):
    @nb.njit
    def weighted_mean_numba(objattr, w, k, dkup, dkdown, addsyst, nweight):
        wsum = np.zeros_like(objattr, dtype=np.float32)
        wksum = np.zeros_like(objattr, dtype=np.float32)
        wdkupsum = np.zeros_like(objattr, dtype=np.float32)
        wdkdownsum = np.zeros_like(objattr, dtype=np.float32)

        for idx in range(objattr.shape[0]):
            for subidx in range(nweight*idx, nweight*(idx+1)):
                wsum[idx] += w[subidx]
                wksum[idx] += w[subidx]*k[subidx]
                wdkupsum[idx] += (w[subidx]*dkup[subidx])**2
                wdkdownsum[idx] += (w[subidx]*dkdown[subidx])**2

        mean = wksum / wsum
        unc_up = np.sqrt((wdkupsum / wsum**2) + addsyst**2)
        unc_down = -1.*np.sqrt((wdkdownsum / wsum**2) + addsyst**2)
        return mean.astype(np.float32), unc_up.astype(np.float32), unc_down.astype(np.float32)

    event_vars = [v(ev, source, nsig) for v in bins_vars]
    for v in event_vars:
        v.content[np.isnan(v.content)] = 0.

    indices = get_bin_indices(
        [event_vars[idx].content.astype(np.float32) for idx in range(len(event_vars))],
        [df["bin{}_low".format(idx)].values.astype(np.float32) for idx in range(len(event_vars))],
        [df["bin{}_upp".format(idx)].values.astype(np.float32) for idx in range(len(event_vars))],
        df["weight"].unique().shape[0],
    ).ravel()
    dfw = df.iloc[indices]

    sf, sfup, sfdown = weighted_mean_numba(
        event_vars[0].content, dfw["weight"].values, dfw["corr"].values,
        dfw["unc_up"].values, dfw["unc_down"].values,
        add_syst(ev, source, nsig).content, df["weight"].unique().shape[0],
    )

    return awk.JaggedArray(
        event_vars[0].starts, event_vars[0].stops,
        weight_numba(sf, nsig, sfup, sfdown),
    )

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

                df = pd.read_csv(path, sep='\s+')
                df.columns = file_form
                df = df.sort_values(file_form).reset_index(drop=True)

                for idx in range(ndims):
                    bin_low = "bin{}_low".format(idx)
                    bin_upp = "bin{}_upp".format(idx)
                    df.loc[df[bin_low]==df[bin_low].min(), bin_low] = -np.inf
                    df.loc[df[bin_upp]==df[bin_upp].max(), bin_upp] =  np.inf

                df["weight"] = w
                dfs.append(df)
                corrector["df"] = pd.concat(dfs).reset_index(drop=True)
        # fin init

    def begin(self, event):
        funcs = [corrector["binning_variables"] for corrector in self.correctors]
        funcs.extend([(corrector["add_syst"],) for corrector in self.correctors])
        self.lambda_functions = {
            func: Lambda(func)
            for func in [f for fs in funcs for f in fs]
        }

        for corrector in self.correctors:
            vname = corrector["name"]
            cname = corrector["collection"]
            event.register_function(
                event,
                "{}_Weight{}SF".format(cname, vname),
                partial(
                    evaluate_object_weights, df=corrector["df"],
                    bins_vars=[self.lambda_functions[v] for v in corrector["binning_variables"]],
                    add_syst=self.lambda_functions[corrector["add_syst"]],
                    name=vname,
                ),
            )

    def end(self):
        self.lambda_functions = None

import numpy as np
import pandas as pd
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.NumbaFuncs import get_bin_indices, weight_numba

def qcdewk_weight(ev, source, nsig, parent, input_paths, variations, input_df):
    weights = {}
    if parent not in input_paths:
        weights[""] = np.ones(ev.size, dtype=np.float32)
        for variation in variations[1:]:
            weights[variation] = np.zeros(ev.size, dtype=np.float32)
    else:
        indices = get_bin_indices(
            [ev.GenPartBoson(ev, 'pt')],
            [input_df.index.get_level_values("bin_min").values],
            [input_df.index.get_level_values("bin_max").values],
            1,
        )[:,0]
        corrections = input_df.iloc[indices]
        weights[""] = corrections[""].values.astype(np.float32)
        for variation in variations[1:]:
            weights[variation] = ((corrections[variation]/corrections[""]).values - 1.).astype(np.float32)

    central = weights[""]
    try:
        up = weights[source+"Up"]
        down = weights[source+"Down"]
    except KeyError:
        up = np.zeros_like(central, dtype=np.float32)
        down = np.zeros_like(central, dtype=np.float32)
    return weight_numba(central, nsig, up, down)

class WeightQcdEwk(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        self.variations = [""]\
                + [n+"Up" for n in self.variation_names]\
                + [n+"Down" for n in self.variation_names]

        self.parent = event.config.dataset.parent
        if self.parent in self.input_paths:
            input_dict = {}
            global_bin_min = None
            global_bin_max = None
            for param in self.params:
                path, key = self.input_paths[self.parent]
                bin_min, bin_max, correction, _, _ = read_input(path, key.format(param))

                if global_bin_min is None:
                    global_bin_min = bin_min
                if global_bin_max is None:
                    global_bin_max = bin_max

                assert np.all(global_bin_min == bin_min)
                assert np.all(global_bin_max == bin_max)

                input_dict[param] = correction

            input_df = pd.DataFrame(input_dict)
            input_df["bin_min"] = global_bin_min
            input_df["bin_max"] = global_bin_max
            input_df = input_df.set_index(["bin_min", "bin_max"])

            input_df["isz"] = self.parent in ["ZJetsToNuNu", "DYJetsToLL", "ZJetsToLL", "GStarJetsToLL"]
            input_df["isw"] = self.parent in ["WJetsToLNu"]

            # nominal
            columns = [""]
            for var in self.variation_names:
                input_df[var] = 0
            input_df[""] = input_df.eval(self.formula)

            # Up/down variations
            for var in self.variation_names:
                input_df[var] = 1
                input_df[var+"Up"] = input_df.eval(self.formula)
                input_df[var] = -1
                input_df[var+"Down"] = input_df.eval(self.formula)
                input_df[var] = 0
                columns.append(var+"Up")
                columns.append(var+"Down")
            input_df = input_df[columns]

            if self.overflow:
                ser = input_df.iloc[-1,:]
                ser.name = (ser.name[-1], np.inf)
                input_df = input_df.append(ser)
            if self.underflow:
                ser = input_df.iloc[0,:]
                ser.name = (-np.inf, ser.name[0])
                input_df = input_df.append(ser)
                input_df.iloc[-1,:] = 1
            self.input_df = input_df.sort_index()
        else:
            self.input_df = None

        # Register function
        event.register_function(
            event, "WeightQcdEwk", partial(
                qcdewk_weight, parent=self.parent, input_paths=self.input_paths,
                variations=self.variations, input_df=self.input_df,
            ),
        )

def read_input(path, histname):
    with open(path, 'r') as f:
        lines = f.read().splitlines()

    start_idx = next(
        idx for idx in range(len(lines))
        if "# BEGIN HISTO1D {}".format(histname) in lines[idx]
    )
    end_idx = next(
        idx for idx in range(start_idx, len(lines))
        if "# END HISTO1D" in lines[idx]
    )

    data = np.array([
        [float(w) for w in l.split()]
        for l in lines[start_idx+2: end_idx]
    ])

    bin_min, bin_max = data[:,0], data[:,1]
    correction = data[:,2]
    errdown, errup = data[:,3], data[:,4]

    return bin_min, bin_max, correction, errdown, errup

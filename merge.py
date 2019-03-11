import argparse
import os
import pandas as pd
import re

try:
    import cPickle as pickle
except ImportError:
    import pickle

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("inputs", type=str, help="Comma delimited input files")
    parser.add_argument("regex", type=str, help="Weights to regex merge on")
    parser.add_argument("-o", "--output", type=str, default="results.pkl",
                        help="File to save the output result")

    return parser.parse_args()

def read_inputs(inputs):
    input_list = []
    for input in inputs:
        with open(input, 'r') as f:
            try:
                input_list.append(pickle.load(f))
            except pickle.UnpicklingError:
                raise IOError("Unpickling error in {}".format(input))
    return input_list

def merge_dfs(dfs, regex):
    weights = dfs[0].index.get_level_values("weight").unique()
    weights_nomatch = [w for w in weights if not regex.search(w)]
    dfs_match = []
    for df in dfs:
        weights = df.index.get_level_values("weight").unique()
        weights_match = [w for w in weights if regex.search(w)]

        df = df.loc[df.index.get_level_values("weight").isin(weights_match)]
        dfs_match.append(df)

    dfs_nomatch = dfs[0].loc[dfs[0].index.get_level_values("weight").isin(weights_nomatch)]
    return pd.concat([dfs_nomatch]+dfs_match, axis=0)

def save(result, output):
    dname = os.path.dirname(output)
    if not os.path.exists(dname) and dname != "":
        os.makedirs(dname)

    with open(output, 'w') as f:
        print(result)
        pickle.dump(result, f)
    print("Created {}".format(output))

def main():
    options = parse_args()
    inputs = read_inputs(options.inputs.split(","))
    for idx in range(len(inputs)-1):
        assert inputs[idx][0] == inputs[idx+1][0]
    binning = inputs[0][0]
    dfs = [i[1] for i in inputs]
    df = merge_dfs(dfs, re.compile(options.regex))
    result = (binning, df)
    save(result, options.output)

if __name__ == "__main__":
    main()

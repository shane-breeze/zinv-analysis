import argparse
import importlib

try: import cPickle as pickle
except ImportError: import pickle

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("inputs", type=str,
                        help="Input pickle files to draw (comma delimited)")
    parser.add_argument("drawer", type=str,
                        help="Drawing function to exectue on the inputs")
    parser.add_argument("-n", "--name", type=str, default="test",
                        help="Name of pdf")

    return parser.parse_args()

def main():
    options = parse_args()

    paths = options.inputs.split(",")

    args = []
    for path in paths:
        with open(path, 'r') as f:
            args.append(pickle.load(f))

    module_name, function_name = options.drawer.split(":")
    draw = getattr(importlib.import_module(module_name), function_name)
    draw(*args, filepath=options.name)

if __name__ == "__main__":
    main()

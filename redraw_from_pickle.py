import argparse
import importlib

try: import cPickle as pickle
except ImportError: import pickle

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str,
                        help="Input pickle file to redraw")
    parser.add_argument("drawer", type=str,
                        help="Drawing function to exectue on the inputs")

    return parser.parse_args()

def main():
    options = parse_args()

    path = options.input
    with open(path, 'r') as f:
        args = pickle.load(f)

    module_name, function_name = options.drawer.split(":")
    draw = getattr(importlib.import_module(module_name), function_name)
    draw(*args)

if __name__ == "__main__":
    main()

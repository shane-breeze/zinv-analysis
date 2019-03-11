import argparse
import importlib
import pickle

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str,
                        help="Input pickle files to draw")
    parser.add_argument("drawer", type=str,
                        help="Drawing function to exectue on the input")

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

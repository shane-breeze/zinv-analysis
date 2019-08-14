#!/usr/bin/env python
import argparse
from zinv.modules import resume

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to recreate")
    parser.add_argument(
        "--sge-opts", type=str, default="-q hep.q -l h_vmem=24G",
        help="SGE options",
    )
    return parser.parse_args()

def main():
    options = parse_args()
    resume(**options)

if __name__ == "__main__":
    main()

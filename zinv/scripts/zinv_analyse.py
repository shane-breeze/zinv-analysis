#!/usr/bin/env python
import os
import sys
import pysge
from cachetools import LFUCache
import warnings
warnings.filterwarnings('ignore')

from atuproot.atuproot_main import AtUproot
from zinv.utils.gittools import git_diff, git_revision_hash
from zinv.utils.cache_funcs import get_size
from zinv.datasets.datasets import get_datasets
from zinv.sequence.config import build_sequence

import logging
logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger("alphatwirl").setLevel(logging.INFO)
logging.getLogger("atsge.SGEJobSubmitter").setLevel(logging.INFO)
logging.getLogger("alphatwirl.progressbar.ProgressReport").setLevel(logging.ERROR)

logging.getLogger(__name__).propagate = False
logging.getLogger("alphatwirl").propagate = False
logging.getLogger("atsge.SGEJobSubmitter").propagate = False
logging.getLogger("atuproot.atuproot_main").propagate = False
logging.getLogger("alphatwirl.progressbar.ProgressReport").propagate = False

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_cfg", type=str,
                        help="Dataset config to run over")
    parser.add_argument("sequence_cfg", type=str,
                        help="Config for how to process events")
    parser.add_argument("event_selection_cfg", type=str,
                        help="Config for the event selection")
    parser.add_argument("physics_object_cfg", type=str,
                        help="Config for the physics object selection")
    parser.add_argument("trigger_cfg", type=str,
                        help="Config for the HLT trigger paths")
    parser.add_argument("hdf_cfg", type=str,
                        help="Config for the output HDF files")
    parser.add_argument("-o", "--outdir", default="output", type=str,
                        help="Where to save the results")
    parser.add_argument("--mode", default="multiprocessing", type=str,
                        help="Which mode to run in (multiprocessing, htcondor, "
                             "sge)")
    parser.add_argument("--sge-opts", type=str,
                        default="-q hep.q -l h_rt=3:0:0 -l h_vmem=24G",
                        help="SGE options")
    parser.add_argument("--ncores", default=0, type=int,
                        help="Number of cores to run on")
    parser.add_argument("--nblocks-per-dataset", default=-1, type=int,
                        help="Number of blocks per dataset")
    parser.add_argument("--nblocks-per-process", default=-1, type=int,
                        help="Number of blocks per process")
    parser.add_argument("--nfiles-per-dataset", default=-1, type=int,
                        help="Number of files per dataset")
    parser.add_argument("--nfiles-per-process", default=1, type=int,
                        help="Number of files per process")
    parser.add_argument("--blocksize", default=1000000, type=int,
                        help="Number of events per block")
    parser.add_argument("--cachesize", default=8*1024**3, type=int,
                        help="Branch cache size")
    parser.add_argument("--quiet", default=False, action='store_true',
                        help="Keep progress report quiet")
    parser.add_argument("--sample", default=None, type=str,
                        help="Select some sample (comma delimited). Can "
                        "selected from (data, mc and more)")
    return parser.parse_args()

def generate_report(outdir):
    # command
    filepath = os.path.join(outdir, "report.txt")
    with open(filepath, 'w') as f:
        f.write("python "+" ".join(sys.argv)+"\n")

    # git hash
    filepath = os.path.join(outdir, "git_hash.txt")
    hash = git_revision_hash()
    with open(filepath, 'w') as f:
        f.write(hash)

    # git diff
    filepath = os.path.join(outdir, "git_diff.txt")
    with open(filepath, 'w') as f:
        f.write(git_diff())

    # commands to checkout the hash with the diffs applied
    string = "#!/bin/bash\n"
    string += "git clone git@github.com:shane-breeze/zinv-analysis.git\n"
    string += "git checkout {}\n".format(hash)
    string += "git apply {}\n".format(os.path.abspath(filepath))
    filepath = os.path.join(outdir, "git_checkout.sh")
    with open(filepath, 'w') as f:
        f.write(string)

def run(sequence, datasets, options):
    process = AtUproot(
        options.outdir,
        quiet = options.quiet,
        max_blocks_per_dataset = options.nblocks_per_dataset,
        max_blocks_per_process = options.nblocks_per_process,
        max_files_per_dataset = options.nfiles_per_dataset,
        max_files_per_process = options.nfiles_per_process,
        nevents_per_block = options.blocksize,
        branch_cache = LFUCache(options.cachesize, get_size),
    )
    tasks = process.run(datasets, sequence)

    if options.mode=="multiprocessing" and options.ncores==0:
        results = pysge.local_submit(tasks)
    elif options.mode=="multiprocessing":
        results = pysge.mp_submit(tasks, ncores=options.ncores)
    elif options.mode=="sge":
        results = pysge.sge_submit(
            "zinv", "_ccsp_temp/", tasks=tasks, options=options.sge_opts,
            sleep=5, request_resubmission_options=True,
        )
    return results

if __name__ == "__main__":
    options = parse_args()
    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)
    generate_report(options.outdir)

    sequence = build_sequence(
        options.sequence_cfg, options.outdir, options.event_selection_cfg,
        options.physics_object_cfg, options.trigger_cfg, options.hdf_cfg,
    )
    datasets = get_datasets(options.dataset_cfg)

    if options.sample is not None:
        if options.sample.lower() == "data":
            datasets = [d for d in datasets
                        if "run2016" in d.name.lower()]
        elif options.sample.lower() == "mc":
            datasets = [d for d in datasets
                        if "run2016" not in d.name.lower()]
        else:
            samples = options.sample.split(",")
            datasets = [d for d in datasets
                        if d.name in samples or d.parent in samples]

    # Pass any other options through to the datasets
    #for d in datasets:
    #    d.systs = options.systs
    results = run(sequence, datasets, options)

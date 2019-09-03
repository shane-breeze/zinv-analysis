#!/usr/bin/env python
from zinv.modules import analyse

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger("alphatwirl").setLevel(logging.INFO)
logging.getLogger("alphatwirl.progressbar.ProgressReport").setLevel(logging.ERROR)

logging.getLogger(__name__).propagate = False
logging.getLogger("alphatwirl").propagate = False
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
    parser.add_argument("-n", "--name", default="zinv", type=str,
                        help="Name to pass to batch")
    parser.add_argument("-o", "--outdir", default="output", type=str,
                        help="Where to save the results")
    parser.add_argument("-t", "--tempdir", default="_ccsp_temp", type=str,
                        help="Where to store the temp directory")
    parser.add_argument("--mode", default="multiprocessing", type=str,
                        help="Which mode to run in (multiprocessing, htcondor, "
                             "sge)")
    parser.add_argument("--batch-opts", type=str,
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
    parser.add_argument("--dryrun", default=False, action='store_true',
                        help="Don't submit the jobs to a batch system")
    parser.add_argument("--sample", default=None, type=str,
                        help="Select some sample (comma delimited). Can "
                        "selected from (data, mc and more)")
    return parser.parse_args()

if __name__ == "__main__":
    analyse(**vars(parse_args()))

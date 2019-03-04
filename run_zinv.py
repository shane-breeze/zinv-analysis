#!/usr/bin/env python
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from atuproot.atuproot_main import AtUproot
from atsge.build_parallel import build_parallel
from utils.grouped_run import grouped_run
from utils.gittools import git_diff, git_revision_hash
from datasets.datasets import get_datasets
from sequence.config import build_sequence

import logging
logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger("alphatwirl").setLevel(logging.INFO)
logging.getLogger("atsge.SGEJobSubmitter").setLevel(logging.INFO)

logging.getLogger(__name__).propagate = False
logging.getLogger("alphatwirl").propagate = False
logging.getLogger("atsge.SGEJobSubmitter").propagate = False
logging.getLogger("atuproot.atuproot_main").propagate = False

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
    parser.add_argument("weight_cfg", type=str,
                        help="Config for the weight sequence")
    parser.add_argument("-o", "--outdir", default="output", type=str,
                        help="Where to save the results")
    parser.add_argument("--mode", default="multiprocessing", type=str,
                        help="Which mode to run in (multiprocessing, htcondor, "
                             "sge)")
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
    parser.add_argument("--quiet", default=False, action='store_true',
                        help="Keep progress report quiet")
    parser.add_argument("--profile", default=False, action='store_true',
                        help="Profile the code")
    parser.add_argument("--sample", default=None, type=str,
                        help="Select some sample (comma delimited). Can "
                        "selected from (data, mc and more)")
    parser.add_argument("--redraw", default=False, action='store_true',
                        help="Overrides most options. Runs over collectors "
                             "only to rerun the draw function on outdir")
    parser.add_argument("--nodraw", default=False, action='store_true',
                        help="Don't run drawing processes")
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

vmem_dict = {
    # 500,000 events per block:
    "DYJetsToLL_Pt-0To50":         16,
    "DYJetsToLL_Pt-50To100":       16,
    "DYJetsToLL_Pt-50To100_ext1":  16,
    "DYJetsToLL_Pt-100To250_ext3": 20,
    "DYJetsToLL_Pt-250To400_ext3": 20,
    "G1Jet_Pt-50To100_ext1":  16,
    "G1Jet_Pt-100To250_ext2": 16,
    "G1Jet_Pt-250To400_ext2": 16,
    "SingleTop_tW_antitop_InclusiveDecays":        20,
    "SingleTop_tW_top_InclusiveDecays":            20,
    "SingleTop_t-channel_top_InclusiveDecays":     20,
    "SingleTop_t-channel_antitop_InclusiveDecays": 16,
    "QCD_Pt-50To80":          16,
    "QCD_Pt-80To120":         16,
    "QCD_Pt-120To170":        16,
    "QCD_Pt-120To170_ext1":   16,
    "QCD_Pt-170To300":        16,
    "QCD_Pt-170To300_ext1":   16,
    "QCD_Pt-300To470":        16,
    "QCD_Pt-300To470_ext1":   20,
    "QCD_Pt-470To600":        16,
    "QCD_Pt-600To800":        16,
    "QCD_Pt-600To800_ext1":   20,
    "QCD_Pt-800To1000":       16,
    "QCD_Pt-800To1000_ext1":  20,
    "QCD_Pt-1000To1400":      16,
    "QCD_Pt-1000To1400_ext1": 20,
    "QCD_Pt-1400To1800_ext1": 16,
    "TTJets_Inclusive": 24,
    "WJetsToLNu_Pt-0To50":         16,
    "WJetsToLNu_Pt-50To100":       16,
    "WJetsToLNu_Pt-100To250":      16,
    "WJetsToLNu_Pt-100To250_ext1": 16,
    "WJetsToLNu_Pt-100To250_ext2": 16,
    "WJetsToLNu_Pt-250To400_ext2": 20,
    "WWTo1L1Nu2Q": 16,
    "WWTo4Q":      16,
    "WZTo1L1Nu2Q":    16,
    "WZTo1L3Nu_ext1": 16,
    "WZTo2L2Q":       16,
    "WZTo2Q2Nu":      16,
    "WZTo3L1Nu":      16,
    "ZJetsToNuNu_Pt-100To250":      16,
    "ZJetsToNuNu_Pt-100To250_ext1": 16,
    "ZJetsToNuNu_Pt-100To250_ext2": 16,
    "ZJetsToNuNu_Pt-250To400_ext2": 16,
    "ZZTo2L2Q":       16,
    "ZZTo2L2Nu":      16,
    "ZZTo2L2Nu_ext1": 16,
    "ZZTo2Q2Nu":      16,
    "ZZTo4L":         16,
    "ZZTo4Q":         16,
    "ZGToLLG": 16,
    "WGToLNuG": 16,
    "EWKWPlusToLNu2Jets_ext2":  16,
}

def run(sequence, datasets, options):
    process = AtUproot(
        options.outdir,
        quiet = options.quiet,
        max_blocks_per_dataset = options.nblocks_per_dataset,
        max_blocks_per_process = options.nblocks_per_process,
        max_files_per_dataset = options.nfiles_per_dataset,
        max_files_per_process = options.nfiles_per_process,
        nevents_per_block = options.blocksize,
        profile = options.profile,
        profile_out_path = "profile.txt",
    )

    # Change parallel options (SGE not supported in standard `build_parallel`)
    process.parallel_mode = options.mode
    process.parallel = build_parallel(
        options.mode,
        quiet = options.quiet,
        processes = options.ncores,
        #user_modules = ('sequence', 'drawing', 'utils'),
        dispatcher_options = {
            "vmem": 12,
            "walltime": 10800,
            "vmem_dict": vmem_dict,
            "walltime_dict": {},
        },
        dropbox_options = {
            "sleep": 10,
        },
    )

    return process.run(datasets, sequence)

def redraw(sequence, datasets, options):
    return [
        collector.reload(options.outdir)
        for (reader, collector) in sequence
        if hasattr(collector, "reload")
    ]

def parallel_draw(jobs, options):
    if len(jobs)==0:
        return
    jobs = [job for subjobs in jobs for job in subjobs]
    jobs = [jobs[i:i+len(jobs)/100+1]
            for i in xrange(0, len(jobs), len(jobs)/100+1)]

    parallel = build_parallel(
        options.mode,
        quiet = options.quiet,
        processes = options.ncores,
        dispatcher_options = {},
    )
    parallel.begin()
    try:
        parallel.communicationChannel.put_multiple([{
            'task': grouped_run,
            'args': args,
            'kwargs': {},
        } for args in jobs])
        parallel.communicationChannel.receive()
    except KeyboardInterrupt:
        parallel.terminate()
    parallel.end()

if __name__ == "__main__":
    options = parse_args()
    if not os.path.exists(options.outdir):
        os.makedirs(options.outdir)
    generate_report(options.outdir)

    sequence = build_sequence(
        options.sequence_cfg, options.outdir, options.event_selection_cfg,
        options.physics_object_cfg, options.trigger_cfg, options.weight_cfg,
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

    if options.redraw:
        jobs = redraw(sequence, datasets, options)
    else:
        jobs = run(sequence, datasets, options)
        if jobs is not None and len(jobs)!=0:
            jobs = [reduce(lambda x, y: x + y, [ssjobs
                for ssjobs in sjobs
                if not ssjobs is None
            ]) for sjobs in jobs]
        else:
            jobs = []
    if not options.nodraw:
        parallel_draw(jobs, options)

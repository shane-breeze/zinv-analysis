import os
import sys
import pysge
from cachetools import LFUCache

from atuproot.atuproot_main import AtUproot
from zinv.utils.gittools import git_diff, git_revision_hash
from zinv.utils.cache_funcs import get_size
from zinv.utils.datasets import get_datasets
from zinv.utils import build_sequence

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

def run(
    sequence, datasets, name, outdir, tempdir, mode, batch_opts, ncores,
    nblocks_per_dataset, nblocks_per_process, nfiles_per_dataset,
    nfiles_per_process, blocksize, cachesize, quiet, sample,
):
    process = AtUproot(
        outdir,
        quiet = quiet,
        max_blocks_per_dataset = nblocks_per_dataset,
        max_blocks_per_process = nblocks_per_process,
        max_files_per_dataset = nfiles_per_dataset,
        max_files_per_process = nfiles_per_process,
        nevents_per_block = blocksize,
        branch_cache = LFUCache(int(cachesize*1024**3), get_size),
    )
    tasks = process.run(datasets, sequence)

    if mode=="multiprocessing" and ncores==0:
        results = pysge.local_submit(tasks)
    elif mode=="multiprocessing":
        results = pysge.mp_submit(tasks, ncores=ncores)
    elif mode=="sge":
        results = pysge.sge_submit(
            name, tempdir, tasks=tasks, options=batch_opts,
            sleep=5, request_resubmission_options=True,
        )
    return results

def analyse(
    dataset_cfg, sequence_cfg, event_selection_cfg, physics_object_cfg,
    trigger_cfg, hdf_cfg, name="zinv", outdir="output", tempdir="_ccsp_temp",
    mode="multiprocessing", batch_opts="-q hep.q", ncores=0,
    nblocks_per_dataset=-1, nblocks_per_process=-1, nfiles_per_dataset=-1,
    nfiles_per_process=1, blocksize=1_000_000, cachesize=8,
    quiet=False, sample=None,
):
    outdir = os.path.abspath(outdir)
    dataset_cfg = os.path.abspath(dataset_cfg)
    sequence_cfg = os.path.abspath(sequence_cfg)
    event_selection_cfg = os.path.abspath(event_selection_cfg)
    physics_object_cfg = os.path.abspath(physics_object_cfg)
    trigger_cfg = os.path.abspath(trigger_cfg)
    hdf_cfg = os.path.abspath(hdf_cfg)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    generate_report(outdir)

    sequence = build_sequence(
        sequence_cfg, outdir, event_selection_cfg, physics_object_cfg,
        trigger_cfg, hdf_cfg,
    )
    datasets = get_datasets(dataset_cfg)

    if sample is not None:
        if sample.lower() == "data":
            datasets = [d for d in datasets
                        if "run2016" in d.name.lower()]
        elif sample.lower() == "mc":
            datasets = [d for d in datasets
                        if "run2016" not in d.name.lower()]
        else:
            samples = sample.split(",")
            datasets = [d for d in datasets
                        if d.name in samples or d.parent in samples]

    # Pass any other options through to the datasets
    #for d in datasets:
    #    pass
    results = run(
        sequence, datasets, name, outdir, tempdir, mode, batch_opts, ncores,
        nblocks_per_dataset, nblocks_per_process, nfiles_per_dataset,
        nfiles_per_process, blocksize, cachesize, quiet, sample,
    )

import os
import inspect
import pysge
from cachetools import LFUCache

from atuproot.atuproot_main import AtUproot
from zinv.utils.gittools import git_diff, git_revision_hash
from zinv.utils.cache_funcs import get_size
from zinv.utils.datasets import get_datasets
from zinv.utils import build_sequence

import numpy as np
import pandas as pd

import logging
logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger("alphatwirl").setLevel(logging.INFO)
logging.getLogger("alphatwirl.progressbar.ProgressReport").setLevel(logging.ERROR)

logging.getLogger(__name__).propagate = False
logging.getLogger("alphatwirl").propagate = False
logging.getLogger("atuproot.atuproot_main").propagate = False
logging.getLogger("alphatwirl.progressbar.ProgressReport").propagate = False

def generate_report(outdir, fname, args, values):
    # command
    filepath = os.path.join(outdir, "report.txt")

    cmd_block = ["{}(".format(fname)]
    for arg in args:
        cmd_block.append("    {} = {},".format(arg, repr(values[arg])))
    cmd_block.append(")")
    with open(filepath, 'w') as f:
        f.write("\n".join(cmd_block)+"\n")

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

def process_results(results, outdir):
    dfs = []
    for res in results:
        df_data = []
        for r in results[0].readers:
            name = r.name
            coll = r.collect()

            for k, v in coll.items():
                df_data.append({
                    "layer": name,
                    "object": k,
                    "time": pd.Timedelta(np.timedelta64(int(v), 'ns')),
                })
                dfs.append(pd.DataFrame(df_data))
    df = pd.concat(dfs).groupby(["layer", "object"]).sum()
    with open(os.path.join(outdir, "timing_report.txt"), 'w') as f:
        f.write(df.sort_values("time", ascending=False).to_string())
    return results

def run(
    sequence, datasets, name, outdir, tempdir, mode, batch_opts, ncores,
    nblocks_per_dataset, nblocks_per_process, nfiles_per_dataset,
    nfiles_per_process, blocksize, cachesize, quiet, dryrun, sample,
    predetermined_nevents_in_file,
):
    process = AtUproot(
        outdir,
        quiet = quiet,
        max_blocks_per_dataset = nblocks_per_dataset,
        max_blocks_per_process = nblocks_per_process,
        max_files_per_dataset = nfiles_per_dataset,
        max_files_per_process = nfiles_per_process,
        nevents_per_block = blocksize,
        predetermined_nevents_in_file=predetermined_nevents_in_file,
        branch_cache = LFUCache(int(cachesize*1024**3), get_size),
    )
    tasks = process.run(datasets, sequence)

    if mode=="multiprocessing" and ncores==0:
        results = pysge.local_submit(tasks)
    elif mode=="multiprocessing":
        results = pysge.mp_submit(tasks, ncores=ncores)
    elif mode=="sge":
        results = pysge.sge_submit(
            tasks, name, tempdir, options=batch_opts, dryrun=dryrun,
            sleep=5, request_resubmission_options=True,
            return_files=True,
        )
    return process_results(results, outdir)

def analyse(
    dataset_cfg, sequence_cfg, event_selection_cfg, physics_object_cfg,
    trigger_cfg, hdf_cfg, name="zinv", outdir="output", tempdir="_ccsp_temp",
    mode="multiprocessing", batch_opts="-q hep.q", ncores=0,
    nblocks_per_dataset=-1, nblocks_per_process=-1, nfiles_per_dataset=-1,
    nfiles_per_process=1, blocksize=1_000_000, cachesize=8,
    quiet=False, dryrun=False, sample=None,
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
        os.makedirs(os.path.join(outdir, "failed"))

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    fname = inspect.getframeinfo(frame)[2]
    generate_report(outdir, fname, args, values)

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

    predetermined_nevents_in_file = {
        f: n
        for d in datasets
        for f, n in zip(d.files, d.file_nevents)
    }

    return run(
        sequence, datasets, name, outdir, tempdir, mode, batch_opts, ncores,
        nblocks_per_dataset, nblocks_per_process, nfiles_per_dataset,
        nfiles_per_process, blocksize, cachesize, quiet, dryrun, sample,
        predetermined_nevents_in_file,
    )

[![CircleCI](https://circleci.com/gh/shane-breeze/zinv-analysis.svg?style=shield)](https://circleci.com/gh/shane-breeze/zinv-analysis)

[![codecov](https://codecov.io/gh/shane-breeze/zinv-analysis/branch/master/graph/badge.svg)](https://codecov.io/gh/shane-breeze/zinv-analysis)

# Z invisible analysis

Setup with

```
source setup.sh
```


To run the analysis use `run_zinv.py`:

```
usage: run_zinv.py [-h] [-o OUTDIR] [--mode MODE] [--ncores NCORES]
                   [--nblocks-per-dataset NBLOCKS_PER_DATASET]
                   [--nblocks-per-process NBLOCKS_PER_PROCESS]
                   [--nfiles-per-dataset NFILES_PER_DATASET]
                   [--nfiles-per-process NFILES_PER_PROCESS]
                   [--blocksize BLOCKSIZE] [--quiet] [--profile]
                   [--sample SAMPLE] [--redraw] [--nodraw]
                   dataset_cfg sequence_cfg event_selection_cfg
                   physics_object_cfg trigger_cfg weight_cfg

positional arguments:
  dataset_cfg           Dataset config to run over
  sequence_cfg          Config for how to process events
  event_selection_cfg   Config for the event selection
  physics_object_cfg    Config for the physics object selection
  trigger_cfg           Config for the HLT trigger paths
  weight_cfg            Config for the weight sequence

optional arguments:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        Where to save the results
  --mode MODE           Which mode to run in (multiprocessing, htcondor, sge)
  --ncores NCORES       Number of cores to run on
  --nblocks-per-dataset NBLOCKS_PER_DATASET
                        Number of blocks per dataset
  --nblocks-per-process NBLOCKS_PER_PROCESS
                        Number of blocks per process
  --nfiles-per-dataset NFILES_PER_DATASET
                        Number of files per dataset
  --nfiles-per-process NFILES_PER_PROCESS
                        Number of files per process
  --blocksize BLOCKSIZE
                        Number of events per block
  --quiet               Keep progress report quiet
  --profile             Profile the code
  --sample SAMPLE       Select some sample (comma delimited). Can selected
                        from (data, mc and more)
  --redraw              Overrides most options. Runs over collectors only to
                        rerun the draw function on outdir
  --nodraw              Don't run drawing processes
```

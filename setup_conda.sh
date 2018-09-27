#!/bin/bash

# setup conda
export PATH="~/miniconda2/bin:$PATH"
source activate py27

# setup root
source /cvmfs/sft.cern.ch/lcg/contrib/gcc/4.8/x86_64-centos7-gcc48-opt/setup.sh
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.14.04/x86_64-centos7-gcc48-opt/root/bin/thisroot.sh

# setup atuproot
export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/atuproot
export TOPDIR=$PWD

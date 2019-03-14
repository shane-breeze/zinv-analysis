#!/bin/bash
#python setup.py install

# setup root
#source /cvmfs/sft.cern.ch/lcg/contrib/gcc/4.8/x86_64-centos7-gcc48-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.14.04/x86_64-centos7-gcc48-opt/root/bin/thisroot.sh

# fix bug with malloc
#export LD_PRELOAD="/usr/lib64/libtcmalloc.so.4"

# setup atuproot
export PYTHONPATH=$PYTHONPATH:$PWD
export TOPDIR=$PWD
export PATH=$PATH:"$TOPDIR/zinv/scripts/"

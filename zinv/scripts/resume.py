#!/usr/bin/env python
# Tai Sakuma <tai.sakuma@gmail.com>
import os, sys
import argparse
import tarfile
import gzip
import cProfile
from io import StringIO, BytesIO
import pstats

try:
    import cPickle as pickle
except:
    import pickle

##__________________________________________________________________||
parser = argparse.ArgumentParser()
parser.add_argument('path', nargs=1, help='path to pickle')
args = parser.parse_args()

##__________________________________________________________________||
def main():

    cwd = os.getcwd()
    workingarea_path = os.path.dirname(args.path[0])
    pickle_basename = os.path.basename(args.path[0])

    setup(cwd, workingarea_path)

    run(cwd, workingarea_path, pickle_basename)

##__________________________________________________________________||
def run(cwd, workingarea_path, pickle_basename):

    os.chdir(workingarea_path)

    with gzip.open(pickle_basename, 'rb') as f:
        reader = pickle.load(f)

    os.chdir(cwd)
    reader.end()

##__________________________________________________________________||
def setup(cwd, workingarea_path):

    os.chdir(workingarea_path)

    dirname = 'python_modules'
    tarname = dirname + '.tar.gz'

    if os.path.exists(tarname) and not os.path.exists(dirname):
        if try_make_file('.untarring'):
            tar = tarfile.open(tarname)
            tar.extractall()
            tar.close()
            os.remove('.untarring')

    while os.path.isfile('.untarring'):
       pass

    if os.path.exists(dirname):
        sys.path.insert(0, dirname)

    os.chdir(cwd)

##__________________________________________________________________||
# http://stackoverflow.com/questions/33223564/atomically-creating-a-file-if-it-doesnt-exist-in-python
def try_make_file(filename):
    try:
        os.open(filename,  os.O_CREAT | os.O_EXCL)
        return True
    except OSError:
        # FileExistsError can be used for Python 3
        return False

##__________________________________________________________________||
def profile_func(func):
    pr = cProfile.Profile()
    pr.enable()
    func()
    pr.disable()
    sortby = 'cumulative'
    try:
        s = StringIO()
        pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby).print_stats()
    except TypeError:
        s = BytesIO()
        pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby).print_stats()
    return s.getvalue()

##__________________________________________________________________||
def print_profile_func(func, profile_out_path=None):
    result = profile_func(func)
    if profile_out_path is None:
        print(result)
    else:
        with open(profile_out_path, 'w') as f:
            f.write(result)

##__________________________________________________________________||
if __name__ == '__main__':
    print_profile_func(
        func = main,
        profile_out_path="/vols/build/cms/sdb15/zinv-analysis/profile.txt",
    )

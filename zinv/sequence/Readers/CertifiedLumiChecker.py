import json
import operator
import numpy as np
import numba as nb
from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

def evaluate_certified_lumi(ev, cert_runs, cert_lumis):
    @nb.njit
    def is_certified_lumi(runs, lumis, cert_runs_, cert_lumis_):
        nev = runs.shape[0]
        is_certified = np.ones(nev, dtype=np.bool8)

        for iev in range(nev):
            # run not in list, skip
            passed = False
            for irun in range(cert_runs_.shape[0]):
                if runs[iev] != cert_runs_[irun]:
                    continue

                cert_lumi_range = cert_lumis_[irun]
                for ibin in range(cert_lumi_range.shape[0]):
                    if cert_lumi_range[ibin,0] <= lumis[iev] <= cert_lumi_range[ibin,1]:
                        passed = True
                        break

                if passed:
                    break
            is_certified[iev] = passed

        return is_certified

    return is_certified_lumi(ev.run, ev.luminosityBlock, cert_runs, cert_lumis)

class CertifiedLumiChecker(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        runs, lumilist = read_json(self.lumi_json_path)
        event.register_function(
            event, 'IsCertified',
            partial(
                evaluate_certified_lumi, cert_runs=runs, cert_lumis=lumilist,
            ),
        )

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    runs = np.array(list(sorted(int(k) for k in data.keys())))
    lumis = [np.array(data[str(r)], dtype=np.int32) for r in runs]
    return runs, lumis

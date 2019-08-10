import numpy as np
import numba as nb
import awkward as awk
import operator
from functools import reduce

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial
from zinv.utils.Geometry import DeltaR2

def evaluate_xclean_mask(ev, source, nsig, obj1name, obj2names, mindr):
    @nb.njit(["boolean[:](float32[:],float32[:],int64[:],int64[:],float32[:],float32[:],int64[:],int64[:])"])
    def xclean_mask_numba(
        etas1, phis1, starts1, stops1, etas2, phis2, starts2, stops2,
    ):
        content = np.ones_like(etas1, dtype=np.bool8)
        for iev, (start1, stop1, start2, stop2) in enumerate(zip(
            starts1, stops1, starts2, stops2,
        )):
            for idx1 in range(start1, stop1):
                eta1 = etas1[idx1]
                phi1 = phis1[idx1]
                for idx2 in range(start2, stop2):
                    deta = eta1 - etas2[idx2]
                    dphi = phi1 - phis2[idx2]

                    # dR**2 < 0.4**2
                    if DeltaR2(deta, dphi) < mindr**2:
                        content[idx1] = False
                        break

        return content

    obj1 = getattr(ev, obj1name)
    masks = []
    for obj2name in obj2names:
        obj2 = getattr(ev, obj2name)
        masks.append(xclean_mask_numba(
            obj1.eta.content, obj1.phi.content,
            obj1.eta.starts, obj1.eta.stops,
            obj2(ev, source, nsig, 'eta').content,
            obj2(ev, source, nsig, 'phi').content,
            obj2(ev, source, nsig, 'eta').starts,
            obj2(ev, source, nsig, 'eta').stops,
        ))
    return awk.JaggedArray(
        obj1.eta.starts, obj1.eta.stops,
        reduce(operator.and_, masks),
    )

class ObjectCrossCleaning(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        for cname in self.collections:
            event.register_function(
                event,
                "{}_XCleanMask".format(cname),
                partial(
                    evaluate_xclean_mask, obj1name=cname,
                    obj2names=self.ref_collections, mindr=self.mindr,
                ),
            )

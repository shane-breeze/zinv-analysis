import numpy as np
import numba as nb
import awkward as awk
import operator

from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial
from utils.Geometry import DeltaR2

def evaluate_xclean_mask(obj1name, obj2names, mindr):
    @nb.njit
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

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_xclean_mask'))
    def fevaluate_xclean_mask(ev, obj1name, obj2names, eidx, nsig, source):
        obj1 = getattr(ev, obj1name)
        masks = []
        for obj2name in obj2names:
            obj2 = getattr(ev, obj2name)
            masks.append(xclean_mask_numba(
                obj1.eta.content, obj1.phi.content,
                obj1.eta.starts, obj1.eta.stops,
                obj2(ev, 'eta').content, obj2(ev, 'phi').content,
                obj2(ev, 'eta').starts, obj2(ev, 'eta').stops,
            ))
        return awk.JaggedArray(
            obj1.eta.starts, obj1.eta.stops,
            reduce(operator.and_, masks),
        )

    def return_evaluate_xclean_mask(ev):
        source = ev.source if ev.source in ev.attribute_variation_sources else ''
        return fevaluate_xclean_mask(ev, obj1name, tuple(obj2names), ev.iblock, ev.nsig, source)

    return return_evaluate_xclean_mask

class ObjectCrossCleaning(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        for cname in self.collections:
            setattr(
                event,
                "{}_XCleanMask".format(cname),
                evaluate_xclean_mask(cname, self.ref_collections, self.mindr),
            )

import numpy as np
import numba as nb
import operator
from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.NumbaFuncs import weight_numba

def evaluate_pdf_variations(valid):
    @nb.njit(["float32[:](float32[:],float32[:],int64[:],int64[:])"])
    def rel_stddev(nominal, pdfs, starts, stops):
        rel_err = np.zeros_like(nominal, dtype=np.float32)
        for iev, (start, stop) in enumerate(zip(starts, stops)):
            if nominal[iev] != 0.:
                rel_err[iev] = np.std(pdfs[start:stop]*nominal[iev])/nominal[iev]
            else:
                rel_err[iev] = 0.
        return rel_err

    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_pdf_variations'))
    def fevaluate_pdf_variations(ev, evidx, nsig, source, valid_):
        if source == "pdf" and valid_:
            pdf_relstddev = rel_stddev(
                ev.LHEWeight_originalXWGTUP,
                ev.LHEPdfWeight.content,
                ev.LHEPdfWeight.starts,
                ev.LHEPdfWeight.stops,
            )
            weight = weight_numba(np.ones(ev.size, dtype=np.float32), nsig, pdf_relstddev, -pdf_relstddev)
        else:
            weight = np.ones(ev.size, dtype=np.float32)
        ev.delete_branches(["LHEWeight_originalXWGTUP", "LHEPdfWeight"])
        return weight

    def ret_func(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ["pdf"]:
            source, nsig = '', 0.
        return fevaluate_pdf_variations(ev, ev.iblock, nsig, source, valid)

    return ret_func

def evaluate_scale_variations(name, positions):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_scale_variations'))
    def fevaluate_scale_variations(ev, evidx, nsig, source, name_):
        if source == name_:
            up = ev.LHEScaleWeight[:,positions[0]]
            down = ev.LHEScaleWeight[:,positions[1]]
            weight = weight_numba(1., nsig, up, down)
        else:
            weight = np.ones(ev.size, dtype=np.float32)
        ev.delete_branches(["LHEScaleWeight"])
        return weight

    def ret_func(ev):
        source, nsig = ev.source, ev.nsig
        if source not in [name]:
            source, nsig = '', 0.
        return fevaluate_scale_variations(ev, ev.iblock, nsig, source, name)

    return ret_func

def evaluate_scale_variations_old(valid):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_scale_variations_old'))
    def fevaluate_scale_variations_old(ev, evidx, nsig, source, valid_):
        if source == "scale" and valid_:
            up = ev.LHEScaleWeight.max(axis=1)
            down = ev.LHEScaleWeight.min(axis=1)
            weight = weight_numba(1., nsig, up, down)
        else:
            weight = np.ones(ev.size, dtype=np.float32)
        ev.delete_branches(["LHEScaleWeight"])
        return weight

    def ret_func(ev):
        source, nsig = ev.source, ev.nsig
        if source not in "scale":
            source, nsig = '', 0.
        return fevaluate_scale_variations_old(ev, ev.iblock, nsig, source, valid)

    return ret_func

class WeightPdfScale(object):
    """
    The mean and variance of the histogrammed PDF variations is equal to the
    sum of the event PDF mean and variance. Use this to our advantage.

    For the scale variation take the following as up/down variations:
        (1, 2) / (1, 0.5)   - muF
        (2, 1) / (0.5, 1)   - muR
        (2, 2) / (0.5, 0.5) - muFxR (i.e. correlation)
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        if event.config.dataset.name in ["TTJets_Inclusive"] or event.config.dataset.parent in ["QCD"]:
            event.WeightPdfVariations = evaluate_pdf_variations(False)
            event.WeightQCDScale = evaluate_scale_variations_old(False)
        else:
            event.WeightPdfVariations = evaluate_pdf_variations(True)
            event.WeightQCDScale = evaluate_scale_variations_old(True)
        #event.WeightFactorScale = evaluate_scale_variations("factor", (5, 3))
        #event.WeightRenormScale = evaluate_scale_variations("renorm", (7, 1))
        #event.WeightFactorXRenormScale = evaluate_scale_variations("factorXrenorm", (8, 0))

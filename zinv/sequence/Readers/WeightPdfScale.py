import re
import numpy as np
import numba as nb
import operator
from cachetools import cachedmethod
from cachetools.keys import hashkey
from functools import partial

from zinv.utils.NumbaFuncs import weight_numba

pdf_regex = re.compile("^pdf(?P<id>[0-9]+)$")
def evaluate_pdf_variations(valid):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_pdf_variations'))
    def fevaluate_pdf_variations(ev, evidx, nsig, source, valid_):
        if valid_:
            nominal = np.ones(ev.size, dtype=np.float32)
            match = pdf_regex.search(source)
            if match:
                pdfid = int(match.group("id"))
                up = ev.LHEPdfWeight[:,pdfid] - 1.
                down = -up
            elif source == "alphas":
                up = ev.LHEPdfWeight[:,101] - 1.
                down = ev.LHEPdfWeight[:,102] - 1.
            else:
                up = np.zeros(ev.size, dtype=np.float32)
                down = -up
            weight = weight_numba(nominal, nsig, up, down)
        else:
            weight = np.ones(ev.size, dtype=np.float32)
        return weight

    def ret_func(ev):
        source, nsig = ev.source, ev.nsig
        match = pdf_regex.search(source)
        if not (match or source=="alphas"):
            source, nsig = '', 0.
        return fevaluate_pdf_variations(ev, ev.iblock, nsig, source, valid)

    return ret_func

def evaluate_scale_variations(valid):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, 'fevaluate_scale_variations'))
    def fevaluate_scale_variations(ev, evidx, nsig, source, valid_):
        if valid_:
            nominal = np.ones(ev.size, dtype=np.float32)
            if source == "muf_scale":
                up = ev.LHEScaleWeight[:,5] - 1.
                down = ev.LHEScaleWeight[:,3] - 1.
            elif source == "mur_scale":
                up = ev.LHEScaleWeight[:,7] - 1.
                down = ev.LHEScaleWeight[:,1] - 1.
            elif source == "mufr_scale":
                up = ev.LHEScaleWeight[:,8] - 1.
                down = ev.LHEScaleWeight[:,0] - 1.
            else:
                up = np.zeros(ev.size, dtype=np.float32)
                down = np.zeros(ev.size, dtype=np.float32)
            weight = weight_numba(nominal, nsig, up, down)
        else:
            weight = np.ones(ev.size, dtype=np.float32)
        return weight

    def ret_func(ev):
        source, nsig = ev.source, ev.nsig
        if source not in ["muf_scale", "mur_scale", "mufr_scale"]:
            source, nsig = '', 0.
        return fevaluate_scale_variations(ev, ev.iblock, nsig, source, valid)

    return ret_func

class WeightPdfScale(object):
    """
    The mean and variance of the histogrammed PDF variations is equal to the
    sum of the event PDF mean and variance. Use this to our advantage.

    For the scale variation take the following as up/down variations:
        (1, 2) / (1, 0.5)   - muF
        (2, 1) / (0.5, 1)   - muR
        (2, 2) / (0.5, 0.5) - muFxR (i.e. correlation)

    LHEScaleWeight[0] -> (0.5, 0.5) # (mur, muf)
                  [1] -> (0.5, 1)
                  [2] -> (0.5, 2)
                  [3] -> (1, 0.5)
                  [4] -> (1, 1)
                  [5] -> (1, 2)
                  [6] -> (2, 0.5)
                  [7] -> (2, 1)
                  [8] -> (2, 2)
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        if event.config.dataset.parent in self.parents_to_skip\
           or event.config.dataset.name in self.parents_to_skip:
            event.WeightPdfVariations = evaluate_pdf_variations(False)
            event.WeightQCDScale = evaluate_scale_variations(False)
        else:
            event.WeightPdfVariations = evaluate_pdf_variations(True)
            event.WeightQCDScale = evaluate_scale_variations(True)

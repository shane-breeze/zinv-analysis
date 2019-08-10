import numpy as np

def evaluate_pdf_variations():
    def ret_func(ev, pos):
        pdf = np.ones(ev.size, dtype=np.float32)
        if ev.hasbranch("LHEPdfWeight"):
            mask = ev.nLHEPdfWeight>pos
            pdf[mask] = ev.LHEPdfWeight[mask,pos]
        return pdf

    return ret_func

def evaluate_scale_variations():
    def ret_func(ev, pos):
        scale = np.ones(ev.size, dtype=np.float32)
        if ev.hasbranch("LHEScaleWeight"):
            mask = ev.nLHEScaleWeight>pos
            scale[mask] = ev.LHEScaleWeight[mask,pos]
        return scale

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
        event._nonbranch_cache["WeightPdfVariations"] = evaluate_pdf_variations()
        event._nonbranch_cache["WeightScaleVariations"] = evaluate_scale_variations()

import numpy as np
from numba import njit, int32, boolean

@njit
def get_val_indices(vals, refs):
    idxs = -1*np.ones_like(vals, dtype=int32)
    for iev, val in enumerate(vals):
        for idx, ref in enumerate(refs):
            if val == ref:
                idxs[iev] = idx
                break
    return idxs

@njit
def get_bin_indices(vals, mins, maxs):
    idxs = -1*np.ones_like(vals, dtype=int32)
    for iev, val in enumerate(vals):
        for ib, (bin_min, bin_max) in enumerate(zip(mins, maxs)):
            if bin_min <= val < bin_max:
                idxs[iev] = ib
                break
    return idxs

@njit
def get_val_mask(vals, refs):
    mask = np.zeros((vals.shape[0], refs.shape[0]), dtype=boolean)
    for iev, val in enumerate(vals):
        for idx, ref in enumerate(refs):
            mask[iev, idx] = (val == ref)
    return mask

@njit
def get_bin_mask(vals, mins, maxs):
    mask = np.zeros((vals.shape[0], mins.shape[0]), dtype=boolean)
    for iev, val in enumerate(vals):
        for ib, (bin_min, bin_max) in enumerate(zip(mins, maxs)):
            mask[iev, ib] = (bin_min <= val < bin_max)
    return mask

@njit
def get_nth_sorted_object_indices(n, pts, starts, stops):
    idxs = -1*np.ones_like(starts, dtype=int32)
    for iev, (start, stop) in enumerate(zip(starts, stops)):
        if n < stop-start:
            idxs[iev] = start + np.argsort(pts[start:stop])[::-1][n]
    return idxs

import numpy as np
from numba import njit, int32

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
def get_nth_sorted_object_indices(n, pts, starts, stops):
    idxs = -1*np.ones_like(starts, dtype=int32)
    for iev, (start, stop) in enumerate(zip(starts, stops)):
        if n < stop-start:
            idxs[iev] = start + np.argsort(pts[start:stop])[::-1][n]
    return idxs

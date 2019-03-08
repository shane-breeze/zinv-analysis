import numpy as np
from numba import njit, boolean, int32, float64

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

@njit
def get_event_object_idx(contents, starts, stops):
    evidx = -1*np.ones_like(contents, dtype=int32)
    obidx = -1*np.ones_like(contents, dtype=int32)
    for idx, (start, stop) in enumerate(zip(starts, stops)):
        evidx[start:stop] = idx
        for subidx in range(start, stop):
            obidx[subidx] = subidx-start
    return evidx, obidx

@njit
def event_to_object_var(variable, starts, stops):
    new_obj_var = np.zeros(stops[-1], dtype=float64)
    for idx, (start, stop) in enumerate(zip(starts, stops)):
        for subidx in range(start, stop):
            new_obj_var[subidx] = variable[idx]
    return new_obj_var

@njit
def interp(x, xp, fp):
    if x < xp[0]:
        return fp[0]
    elif x > xp[-1]:
        return fp[-1]

    for ix in range(xp.shape[0]-1):
        if xp[ix] <= x < xp[ix+1]:
            return (x - xp[ix]) * (fp[ix+1] - fp[ix]) / (xp[ix+1] - xp[ix]) + fp[ix]
    return np.nan

@njit
def interpolate(x, xp, fp):
    result = np.zeros_like(x, dtype=float64)
    for idx in range(x.shape[0]):
        result[idx] = interp(x[idx], xp[idx,:], fp[idx,:])
    return result

@njit
def index_nonzero(x, size):
    result = np.zeros((x.shape[0], size), dtype=int32)
    for iev in range(x.shape[0]):
        pos = 0
        for idx in range(x.shape[1]):
            if x[iev,idx]:
                assert pos < size
                result[iev,pos] = idx
                pos += 1
    return result

@njit
def weight_numba(nominal, nsig, up, down):
    return nominal * (1 + (nsig>=0)*nsig*up - (nsig<0)*nsig*down)

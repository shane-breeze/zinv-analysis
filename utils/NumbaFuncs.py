import numpy as np
from numba import njit, boolean, int32, float32, float64

@njit
def get_bin_indices(vars, mins, maxs, size):
    result = -1*np.ones((vars[0].shape[0], size), dtype=int32)
    for iev in range(vars[0].shape[0]):
        pos = 0
        for ib in range(mins[0].shape[0]):
            accept = True
            for idx in range(len(vars)):
                if not (mins[idx][ib] <= vars[idx][iev] < maxs[idx][ib]):
                    accept = False

            if accept:
                assert pos < size
                result[iev,pos] = ib
                pos += 1

        #assert pos == size

    return result

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
    new_obj_var = np.zeros(stops[-1], dtype=float32)
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
def weight_numba(nominal, nsig, up, down):
    return nominal * (1 + (nsig>=0)*nsig*up - (nsig<0)*nsig*down)

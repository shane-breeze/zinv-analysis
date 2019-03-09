import numpy as np
import numba as nb

def get_bin_indices(vars, mins, maxs, size):
    @nb.njit
    def get_multibin_indices(vars_, mins_, maxs_, size_):
        result = -1*np.ones((vars_[0].shape[0], size_), dtype=nb.int32)
        for iev in range(vars_[0].shape[0]):
            pos = 0
            for ib in range(mins_[0].shape[0]):
                accept = True
                for idx in range(len(vars_)):
                    if not (mins_[idx][ib] <= vars_[idx][iev] < maxs_[idx][ib]):
                        accept = False

                if accept:
                    assert pos < size_
                    result[iev,pos] = ib
                    pos += 1

            #assert pos == size_

        return result

    @nb.njit
    def get_1dbin_indices(var_, min_, max_, size_):
        result = -1*np.ones((var_.shape[0], size_), dtype=nb.int32)
        for iev in range(var_.shape[0]):
            pos = 0
            for ib in range(min_.shape[0]):
                if min_[ib] <= var_[iev] < max_[ib]:
                    assert pos < size_
                    result[iev,pos] = ib
                    pos += 1

            # assert pos == size_

        return result

    if len(vars) == 1:
        return get_1dbin_indices(vars[0], mins[0], maxs[0], size)
    else:
        return get_multibin_indices(vars, mins, maxs, size)

@nb.njit
def get_nth_sorted_object_indices(n, pts, starts, stops):
    idxs = -1*np.ones_like(starts, dtype=nb.int32)
    for iev, (start, stop) in enumerate(zip(starts, stops)):
        if n < stop-start:
            idxs[iev] = start + np.argsort(pts[start:stop])[::-1][n]
    return idxs

@nb.njit
def get_event_object_idx(contents, starts, stops):
    evidx = -1*np.ones_like(contents, dtype=nb.int32)
    obidx = -1*np.ones_like(contents, dtype=nb.int32)
    for idx, (start, stop) in enumerate(zip(starts, stops)):
        evidx[start:stop] = idx
        for subidx in range(start, stop):
            obidx[subidx] = subidx-start
    return evidx, obidx

@nb.njit
def event_to_object_var(variable, starts, stops):
    new_obj_var = np.zeros(stops[-1], dtype=nb.float32)
    for idx, (start, stop) in enumerate(zip(starts, stops)):
        for subidx in range(start, stop):
            new_obj_var[subidx] = variable[idx]
    return new_obj_var

@nb.njit
def interp(x, xp, fp):
    if x < xp[0]:
        return fp[0]
    elif x > xp[-1]:
        return fp[-1]

    for ix in range(xp.shape[0]-1):
        if xp[ix] <= x < xp[ix+1]:
            return (x - xp[ix]) * (fp[ix+1] - fp[ix]) / (xp[ix+1] - xp[ix]) + fp[ix]
    return np.nan

@nb.njit
def interpolate(x, xp, fp):
    result = np.zeros_like(x, dtype=np.float64)
    for idx in range(x.shape[0]):
        result[idx] = interp(x[idx], xp[idx,:], fp[idx,:])
    return result

@nb.njit
def weight_numba(nominal, nsig, up, down):
    return nominal * (1 + (nsig>=0)*nsig*up - (nsig<0)*nsig*down)

# This is slow for some reason!?
@nb.njit
def histogramdd_numba(event_attrs, mins, maxs, weights):
    ndim = len(event_attrs)
    nev = event_attrs[0].shape[0]
    nib = mins[0].shape[0]
    hist = np.zeros(nib, dtype=nb.float32)

    for iev in range(nev):
        for ib in range(nib):
            accept = True
            for idx in range(ndim):
                if not (mins[idx][ib] <= event_attrs[idx][iev] < maxs[idx][ib]):
                    accept = False

            if accept:
                hist[ib] += weights[iev]

    return hist

@nb.njit
def histogram1d_numba(attr, min_, max_, weights):
    hist = np.zeros(min_.shape[0], dtype=nb.float32)

    for iev, x in enumerate(attr):
        for ib, (mn, mx) in enumerate(zip(min_, max_)):
            if mn <= x < mx:
                hist[ib] += weights[iev]
                break

    return hist

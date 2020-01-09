import numba as nb
import numpy as np

@nb.njit
def mtw(ptprod, dphi):
    return np.sqrt(2*ptprod*(1-np.cos(dphi)))

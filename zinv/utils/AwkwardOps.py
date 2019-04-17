import numpy as np
import numba as nb

def get_nth_object(array, id_, ev_size):
    new_array = np.full(ev_size, np.nan)
    new_array[array.count()>id_] = array[array.count()>id_][:,id_]
    return new_array

def get_attr_for_min_ref(array, refarray, ev_size):
    new_array = np.full(ev_size, np.nan)
    array_refmin = array[refarray.argmin()]
    new_array[array_refmin.count()==1] = array_refmin[array_refmin.count()==1]
    return new_array


def jagged_prod(jagged_array):
    @nb.njit(["float32[:](float32[:],int64[:],int64[:])"])
    def jagged_prod_numba(contents, starts, ends):
        prod = np.ones_like(starts, dtype=np.float32)
        for iev, (start, end) in enumerate(zip(starts, ends)):
            for pos in range(start, end):
                prod[iev] *= contents[pos]

        return prod
    return jagged_prod_numba(
        jagged_array.content, jagged_array.starts, jagged_array.stops,
    )

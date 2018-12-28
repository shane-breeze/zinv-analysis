import numpy as np

def get_nth_object(array, id_, ev_size):
    new_array = np.full(ev_size, np.nan)
    new_array[array.count()>id_] = array[array.count()>id_][:,id_]
    return new_array

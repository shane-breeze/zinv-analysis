import awkward as awk

def get_size(arr):
    if isinstance(arr, awk.JaggedArray):
        return arr.content.nbytes + arr.starts.nbytes + arr.stops.nbytes
    return arr.nbytes

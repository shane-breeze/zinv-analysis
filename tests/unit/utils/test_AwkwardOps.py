import pytest
import mock
import numpy as np
import awkward as awk

from zinv.utils.AwkwardOps import (
    get_nth_object,
    get_nth_sorted_object_indices,
    get_attr_for_min_ref,
    jagged_prod,
)

@pytest.mark.parametrize("array,id,size,out", ([
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    0, 3,
    np.array([0, 3, 5]),
], [
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    1, 3,
    np.array([1, 4, 6]),
], [
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    2, 3,
    np.array([2, np.nan, 7]),
], [
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    3, 3,
    np.array([np.nan, np.nan, 8]),
], [
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    4, 3,
    np.array([np.nan, np.nan, np.nan]),
]))
def test_get_nth_object(array, id, size, out):
    assert np.allclose(get_nth_object(array, id, size), out, rtol=1e-5, equal_nan=True)

@pytest.mark.parametrize("array,ref,id,size,out", ([
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    awk.JaggedArray.fromiter([[3, 1, 2], [1, 2], [4, 1, 3, 2]]),
    0, 3,
    np.array([0, 4, 5]),
], [
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    awk.JaggedArray.fromiter([[3, 1, 2], [1, 2], [4, 1, 3, 2]]),
    1, 3,
    np.array([2, 3, 7]),
], [
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    awk.JaggedArray.fromiter([[3, 1, 2], [1, 2], [4, 1, 3, 2]]),
    2, 3,
    np.array([1, np.nan, 8]),
], [
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    awk.JaggedArray.fromiter([[3, 1, 2], [1, 2], [4, 1, 3, 2]]),
    3, 3,
    np.array([np.nan, np.nan, 6]),
], [
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    awk.JaggedArray.fromiter([[3, 1, 2], [1, 2], [4, 1, 3, 2]]),
    4, 3,
    np.array([np.nan, np.nan, np.nan]),
]))
def test_get_nth_sorted_object_indices(array, ref, id, size, out):
    assert np.allclose(get_nth_sorted_object_indices(array, ref, id, size), out, rtol=1e-5, equal_nan=True)

@pytest.mark.parametrize("array,ref,size,out", ([
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]),
    awk.JaggedArray.fromiter([[3, 1, 2], [1, 2], [4, 1, 3, 2]]),
    3,
    np.array([1, 3, 6]),
    ],))
def test_get_attr_for_min_ref(array, ref, size, out):
    assert np.allclose(get_attr_for_min_ref(array, ref, size), out, rtol=1e-5, equal_nan=True)

@pytest.mark.parametrize("input_,output", ([
    awk.JaggedArray.fromiter([[0, 1, 2], [3, 4], [5, 6, 7, 8]]).astype(np.float32),
    np.array([0, 12, 1680]),
],))
def test_jagged_prod(input_, output):
    assert np.allclose(jagged_prod(input_), output, rtol=1e-5, equal_nan=True)

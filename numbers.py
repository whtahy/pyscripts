# Arrays
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from typing import Union, Iterable
    from numpy import ndarray


# https://stackoverflow.com/a/11146645
def cartesian(
        *arrays: 'ndarray') \
        -> 'ndarray':
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype = dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def numpy_info(
        arr: 'ndarray',
        col_width = 40) \
        -> None:
    otype = str(type(arr))
    dtype = arr.dtype.name
    memory = arr.data.nbytes // 1e6
    size = arr.size
    dims = arr.shape
    n_rows = dims[0]
    n_cols = arr.T.shape[0]
    shape = f'{n_rows} x {n_cols}'
    print(f'Dtype  {dtype:>{col_width}}')
    print(f'Size   {size:>{col_width}}')
    print(f'Shape  {shape:>{col_width}}')
    print(f'Type   {otype:>{col_width}}')
    print(f'Memory {memory:>{col_width-3}} MB')


def pslice(
        arr: 'ndarray',
        l_subset: int = 5000) \
        -> 'ndarray':
    n_percents = min(len(arr), l_subset - 1)
    return numpy.percentile(arr,
                            *seq(start = 0,
                                 end = 100,
                                 size = n_percents))


def seq(
        start: 'Union[int, float]',
        end: 'Union[int, float]',
        step: 'Union[int, float]' = 1,
        size: int = None,
        dtype: str = None,
        exclude: 'Iterable[int]' = None,
        incl: bool = True) \
        -> 'ndarray':
    if exclude is None:
        exclude = []
    if start == end:
        return numpy.array([start])
    if size is not None:
        step = numpy.absolute(end - start) / size
    if dtype is None and type(step) is int:
        dtype = 'int'
    arr = numpy.arange(start = start,
                       stop = end + incl - 1,
                       step = step,
                       dtype = dtype)
    endpoint = (arr[-1] + step).astype(dtype)
    if abs(endpoint) <= abs(end):
        arr = numpy.append(arr, endpoint)
    return arr[~numpy.in1d(arr, exclude)]


def consec_sum(
        start: int,
        end: int) \
        -> int:
    return triangle_n(end) - triangle_n(start - 1)


def triangle_n(
        end: int) \
        -> int:
    return int(end * (end + 1) / 2)

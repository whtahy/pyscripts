# Arrays
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

import numpy
from numpy import absolute as abs, sign, floor
from knot import printf

if TYPE_CHECKING:
    from typing import Iterable
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


def consec_sum(
        start: int,
        end: int) \
        -> int:
    return triangle_num(end) - triangle_num(start - 1)


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
    print(f'Memory {f"{memory} MB":>{col_width}}')


def pslice(
        arr: 'ndarray',
        l_subset: int = 5000) \
        -> 'ndarray':
    n_percents = min(len(arr), l_subset - 1)
    return numpy.percentile(arr,
                            *seq(start = 0,
                                 end = 100,
                                 n_elements = n_percents))


def seq(
        start: float,
        end: float,
        step: float = None,
        n_elements: int = None,
        dtype: str = None,
        exclude_vals: 'Iterable[float]' = None,
        exclude_idx: 'Iterable[int]' = None,
        incl: bool = True,
        debug = False) \
        -> 'ndarray':
    if exclude_vals is None:
        exclude_vals = []
    if exclude_idx is None:
        exclude_idx = []

    delta = end - start
    if step is None:
        step = sign(delta)

    if step == 0 or abs(delta) < abs(step):
        return numpy.array([start], dtype = numpy.result_type(start, end, step))
    else:
        if n_elements is None:
            n_elements = 1 + floor(abs(delta) / abs(step))
        else:
            step = delta / n_elements

        if dtype is None:
            dtype = numpy.result_type(start, end, step)

        if debug:
            printf(f'start: {start:<10}')
            printf(f'start: {start:<10}')
            printf(f'start:      {start:<10})end: {end}   dtype: {dtype}')
            print(f'n_elements: {n_elements}delta: {delta}   step : {step}')

        if n_elements % step == 1:
            return numpy.arange(start = start,
                                stop = end + incl * step,
                                dtype = dtype)
        else:
            return numpy.linspace(start = start,
                                  stop = end,
                                  num = n_elements,
                                  dtype = dtype)


def triangle_num(
        n: int) \
        -> int:
    return int(n * (n + 1) / 2)

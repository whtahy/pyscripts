# Arrays
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

import numpy
from numpy import absolute as abs, round, sign

if TYPE_CHECKING:
    from typing import *
    from pyscripts.types import *


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


def is_integer(
        numbers: 'Iterable[float]') \
        -> 'ndarray':
    return numpy.fromiter(map(lambda x: x % 1 == 0, numbers), dtype = 'bool')


def numpy_info(
        arr: 'ndarray',
        col_width: int = 40) \
        -> None:
    otype = str(type(arr))
    dtype_name = arr.dtype.name
    memory = arr.data.nbytes // 1e6
    size = arr.size
    dims = arr.shape
    n_rows = dims[0]
    n_cols = arr.T.shape[0]
    shape = f'{n_rows} x {n_cols}'
    print(f'Dtype  {dtype_name:>{col_width}}')
    print(f'Size   {size:>{col_width}}')
    print(f'Shape  {shape:>{col_width}}')
    print(f'Type   {otype:>{col_width}}')
    print(f'Memory {f"{memory} MB":>{col_width}}')


def pslice(
        arr: 'Iterable[float]',
        percents: 'NpFloatType' = None,
        n_elements: int = 101) \
        -> 'Iterable[float]':
    if percents is None:
        percents = seq(0, 100, n_elements = n_elements)
        return numpy.percentile(arr, percents)
    else:
        if n_elements > len(arr):
            return arr
        else:
            return numpy.percentile(arr, percents)


def pslice_top(
        scores: 'Iterable[float]',
        percents: 'NpFloatType' = None,
        higher_is_better: bool = True) \
        -> 'Iterable[float]':
    if percents is None:
        percents = seq(100, 1)
    if not higher_is_better:
        percents = numpy.array(100) - percents
    return pslice(scores, percents = percents)


# REFACTOR
# refactor: debug -> use table
# impl: exclude_vals
# impl: exclude_idx
def seq(
        start: float,
        end: float,
        step: float = None,
        n_elements: int = None,
        dtype_name: str = None,
        exclude_vals: 'Iterable[float]' = None,
        exclude_idx: 'Iterable[int]' = None,
        incl: bool = True,
        debug = False,
        debug_width = 6) \
        -> 'ndarray':
    if exclude_vals is None:
        exclude_vals = []
    if exclude_idx is None:
        exclude_idx = []

    delta = end - start
    if step is None:
        step = sign(delta)

    if step == 0 or abs(delta) < abs(step) or sign(step) != sign(delta):
        if debug:
            print('--- hard code ---')
        return numpy.array([start], dtype = numpy.result_type(start, end, step))
    else:
        if n_elements is not None:
            step = delta / (n_elements - 1)
            if step % 1 == 0:
                step = int(step)
        else:
            n_elements = 1 + round(abs(delta) / abs(step)).astype('int')

        if dtype_name is None:
            if start is int and end is int and step is int:
                dtype_name = 'int'
            else:
                dtype_name = 'float'

        if debug:
            arange_end = end + incl * sign(step) * (delta % step == 0)
            print(f'start:      {start:>{debug_width}}')
            print(f'end:        {end:>{debug_width}}')
            print(f'arange_end: {arange_end:>{debug_width}}')
            print(f'delta:      {delta:>{debug_width}}')
            print(f'step:       {step:>{debug_width}}')
            print(f'n_elements: {n_elements:>{debug_width}}')
            print(f'dtype:      {dtype_name:>{debug_width}}')

        if type(step) is int:
            adjust = incl * sign(step) * (delta % step == 0)
            if debug:
                print('--- arange ---')
            return numpy.arange(start = start,
                                stop = end + adjust,
                                step = step,
                                dtype = dtype_name)
        else:
            if debug:
                print('--- linspace ---')
            return numpy.linspace(start = start,
                                  stop = end,
                                  num = n_elements,
                                  dtype = dtype_name)


def triangle_num(
        n: int) \
        -> int:
    return int(n * (n + 1) / 2)

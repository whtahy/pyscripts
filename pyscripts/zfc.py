# Numpy extensions
# Released under CC0:
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import itertools
from functools import partial
from typing import TYPE_CHECKING

import numpy
import numpy.core.defchararray as np_char
import numpy.polynomial.polynomial as poly
import pandas
from numpy import absolute as abs, floor, sign
from scipy.special import comb

from pyscripts.hero import re_typename

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *


def cartesian(
        *arrays: 'ndarray') \
        -> 'ndarray':
    # Credit to: https://stackoverflow.com/a/11146645
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


def describe(
        numpy_array: numpy.ndarray) \
        -> 'pandas.DataFrame':
    return pandas.DataFrame(numpy_array).describe()


def flex_apply(
        func: 'Callable',
        flextype_arg: 'FlexType',
        dtype_name: None) \
        -> 'ndarray':
    if is_NLT(flextype_arg):
        return numpy.array([func(n) for n in flextype_arg], dtype = dtype_name)
    else:
        return numpy.array([func(flextype_arg)], dtype = dtype_name)


def islastcol(
        i_col: int,
        arr: 'NLT_Type') \
        -> bool:
    # For use with partial.
    return i_col == numpy_ncols(arr) - 1


def islastrow(
        i_row: int,
        arr: 'NLT_Type') \
        -> bool:
    # For use with partial.
    return i_row == len(arr) - 1


def is_NLT(
        x: 'Any') \
        -> bool:
    return type(x).__name__ in ['ndarray', 'list', 'tuple']


def numpy_hstack(
        *arrs: 'NLT_Type') \
        -> 'ndarray':
    return numpy.hstack([*arrs])


def numpy_info(
        arr: 'ndarray',
        col_width: int = 15) \
        -> None:
    otype_name = type(arr).__name__
    dtype_name = arr.dtype.name
    size = arr.size
    memory = arr.data.nbytes / 1e6
    n_rows = arr.shape[0]
    n_cols = numpy_ncols(arr)
    shape = f'{n_rows} x {n_cols}'
    print(f'Type   {otype_name:>{col_width}}')
    print(f'Dtype  {dtype_name:>{col_width}}')
    print(f'Size   {size:>{col_width}}')
    print(f'Shape  {shape:>{col_width}}')
    print(f'Memory {memory:>{col_width - 3}.1f} MB')


def numpy_ncols(
        arr: 'ndarray') \
        -> int:
    if arr.ndim == 0:
        return None
    elif arr.ndim == 1:
        return 1
    else:
        return arr.shape[1]


def numpy_vstack(
        *arrs: 'NLT_Type') \
        -> 'ndarray':
    return numpy.vstack([*arrs])


def overflow_check(
        *n: float) \
        -> str:
    if numpy.any(abs(n) > numpy.iinfo('int32').max):
        return 'int64'
    else:
        return 'int32'


def pairtable(
        arr: 'ndarray') \
        -> 'ndarray':
    n_rc = triangle_num_inv(len(arr))
    table = numpy.empty((n_rc, n_rc), dtype = 'object')
    r, c = numpy.triu_indices(n_rc)
    coords = numpy.vstack((r, c)).T
    for i, (j, k) in enumerate(coords):
        table[j, k] = arr[i]
    return table


def pslice(
        arr: 'NLT_FloatType',
        percents: 'FlexFloatType' = None,
        n_elements: int = 101) \
        -> 'ndarray':
    if percents is None:
        percents = seq(0, 100, n_elements = n_elements)
        return numpy.percentile(arr, percents)
    else:
        if n_elements > len(arr):
            return numpy.array(arr)
        else:
            return numpy.percentile(arr, percents)


def pslice_top(
        scores: 'NLT_FloatType',
        percents: 'NLT_FloatType' = None,
        higher_is_better: bool = True) \
        -> 'ndarray':
    if percents is None:
        percents = seq(100, 1)
    if not higher_is_better:
        percents = numpy.array(100) - percents
    return pslice(scores, percents = percents)


# TODO: impl exclude by val
# TODO: impl exclude by idx
def seq(
        start: float,
        end: float = None,
        step: float = None,
        incl: bool = True,
        n_elements: int = None,
        dtype_name: str = None,
        exclude_vals: 'NLT_FloatType' = None,
        exclude_idx: 'NLT_IntType' = None,
        debug = False,
        debug_width = 8) \
        -> 'ndarray':
    if start is not None and end is None:
        end = start
        start = 0
    if exclude_vals is None:
        exclude_vals = []
    if exclude_idx is None:
        exclude_idx = []

    delta = end - start

    if step is None:
        if n_elements is None:
            step = int(sign(delta))
        else:
            step = delta / (n_elements - 1)

    if step == 0 or abs(step) > abs(delta) or sign(step) != sign(delta):
        if debug:
            print('--- hard code ---')
        return numpy.array([start], dtype = numpy.result_type(start, step))
    else:
        if dtype_name is None:
            if typename_of(start) == 'int' and typename_of(step) == 'int':
                dtype_name = 'int'
            else:
                dtype_name = 'float'

        if n_elements is None and typename_of(step) != 'float':
            arange_end = end + incl * sign(step) * (delta % step == 0)
            if debug:
                print(f'{"--- arange ---":^{12 + debug_width}}')
                print()
                print(f'start:      {start:>{debug_width}}')
                print(f'end:        {end:>{debug_width}}')
                print(f'arange_end: {arange_end:>{debug_width}}')
                print()
                print(f'delta:      {delta:>{debug_width}}')
                print(f'step:       {step:>{debug_width}}')
                print()
                print(f'start:      {type(start).__name__:>{debug_width}}')
                print(f'end:        {type(end).__name__:>{debug_width}}')
                print(f'step:       {type(step).__name__:>{debug_width}}')
                print(f'dtype:      {dtype_name:>{debug_width}}')
            return numpy.arange(start = start,
                                stop = arange_end,
                                step = step,
                                dtype = dtype_name)
        else:
            if n_elements is None:
                n_steps = floor(delta / step)
                end = start + n_steps * step
                n_elements = n_steps + 1
            if debug:
                print(f'{"--- linspace ---":^{12 + debug_width}}')
                print()
                print(f'start:      {start:>{debug_width}}')
                print(f'end:        {end:>{debug_width}}')
                print()
                print(f'delta:      {delta:>{debug_width}}')
                print(f'step:       {step:>{debug_width}.5f}')
                print(f'n_elements: {n_elements:>{debug_width}}')
                print()
                print(f'start:      {type(start).__name__:>{debug_width}}')
                print(f'end:        {type(end).__name__:>{debug_width}}')
                print(f'step:       {type(step).__name__:>{debug_width}}')
                print(f'dtype:      {dtype_name:>{debug_width}}')
            return numpy.linspace(start = start,
                                  stop = end,
                                  num = n_elements,
                                  dtype = dtype_name)


def stralign_arr(
        left: 'NLT_StrType',
        right: 'NLT_StrType',
        maxlen_right: int = -1) \
        -> 'Tuple[ndarray, int]':
    veclen = numpy.vectorize(len)
    l_left = max(veclen(left))
    l_right = max(veclen(right))

    def pad_left(string, width):
        return string.rjust(width)

    def pad_right(string, width):
        return string.ljust(width)[0:maxlen_right]

    f_left = partial(pad_left, width = l_left)
    f_right = partial(pad_right, width = l_right)
    padded_left = numpy.vectorize(f_left)(left)
    padded_right = numpy.vectorize(f_right)(right)
    out = np_char.add(padded_left, padded_right)

    if maxlen_right > -1:
        l_right = maxlen_right
    return out, l_left + l_right


def strsplit_arr(
        arr: 'NLT_StrType',
        by: str) \
        -> 'Tuple[ndarray, ndarray]':
    def strsplit(s):
        leftright = s.split(by, maxsplit = 1)
        left = leftright[0]
        if len(leftright) == 2:
            right = by + leftright[1]
        else:
            right = ''
        return left, right

    left_arr, right_arr = numpy.vectorize(strsplit)(arr)
    return left_arr, right_arr


def subset_apply(
        func: 'Callable[[Any, Any], Any]',
        arr: 'NLT_Type',
        l_subset: int = 2,
        dtype_name = None) \
        -> 'ndarray':
    out = numpy.empty(comb(len(arr), l_subset).astype('int'),
                      dtype = dtype_name)
    for i, subset in enumerate(itertools.combinations(arr, l_subset)):
        out[i] = func(*subset)
    return out


def subdict(
        superdict: dict,
        keys: 'FlexType') \
        -> dict:
    out = {k: superdict[k] for k in
           numpy.intersect1d(list(superdict.keys()), keys)}
    return out


def triangle_num(
        n: int) \
        -> int:
    return int(n * (n + 1) / 2)


def triangle_num_inv(
        n: int) \
        -> 'Union[int, None]':
    coeff = (2 * n, -1, -1)
    root = max(poly.polyroots(coeff))
    n_inv = int(round(root))
    if triangle_num(n_inv) == n:
        return n_inv
    else:
        return None


def type_map(
        obj: 'Any',
        type_names: 'NLT_StrType',
        codomain: 'NLT_Type',
        default_out: 'Any' = None,
        ignore_bytesize: bool = True) \
        -> 'Any':
    typename_x = typename_of(obj)
    if ignore_bytesize:
        typename_x = re_typename.search(typename_x).group(1)
    for i, typename in enumerate(type_names):
        if typename_x == typename:
            return codomain[i]
    return default_out


def typename_of(
        obj: 'Any',
        ignore_bytesize: bool = True) \
        -> 'Union[str, Tuple[str, str]]':
    # Format: (type)(bytesize)
    name = type(obj).__name__
    if ignore_bytesize:
        return re_typename.search(name).group(1)
    else:
        return re_typename.search(name).group(1), \
               re_typename.search(name).group(2)


def weave(
        arr_left: 'NLT_Type',
        arr_right: 'NLT_Type') \
        -> 'ndarray':
    n_left = len(arr_left)
    n_right = len(arr_right)
    n_weave = min(n_left, n_right)
    out = numpy.empty(n_weave * 2, dtype = 'object')
    out[0: n_weave * 2: 2] = arr_left[0:n_weave]
    out[1: n_weave * 2 + 1: 2] = arr_right[0:n_weave]
    if n_left > n_right:
        leftovers = arr_left[n_right:]
    elif n_left < n_right:
        leftovers = arr_right[n_left:]
    else:
        leftovers = None
    return numpy_hstack(out, leftovers)

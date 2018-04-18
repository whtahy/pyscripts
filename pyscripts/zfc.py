# Numpy extensions
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt

from typing import List, Sequence, Union

import numpy as np
from numpy.polynomial.polynomial import polyroots

from pyscripts.mytypes import Numeric


def cartesian(
        *arrays: Sequence[Numeric]) \
        -> np.ndarray:
    """
    Cartesian product.

    Credit: https://stackoverflow.com/users/577088/senderle
    Source: https://stackoverflow.com/a/11146645
    License: CC BY-SA 3.0
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def interleave(
        left: Sequence,
        right: Sequence) \
        -> List:
    """
    Interleave 2 sequences.
    """
    n_left = len(left)
    n_right = len(right)
    m = min(n_left, n_right)
    n = m * 2

    weave = [None] * n
    weave[0:n:2] = left[0:m]
    weave[1:n + 1:2] = right[0:m]

    if n_left < n_right:
        return weave + right[n_left:]
    elif n_left > n_right:
        return weave + left[n_right:]
    else:
        return weave


def is_int(
        x: Numeric) \
        -> bool:
    """
    Whether `x` is int.

    """
    return isinstance(x, (int, np.integer))


def is_float(
        x: Numeric) \
        -> bool:
    """
    Whether `x` is float.

    """
    return isinstance(x, (float, np.floating))


def n_dec(
        x: Numeric) \
        -> int:
    """
    Number of decimal places, using `str` conversion.

    """
    if x == 0:
        return 0
    _, _, dec = str(x).partition('.')
    return len(dec)


def np_hstack(
        *arrs: Sequence) \
        -> np.ndarray:
    """
    `np.hstack` wrapper.
    """
    return np.hstack([*arrs])


def np_info(
        arr: np.ndarray) \
        -> dict:
    """
    Dict of:
    - `type`
    - `dtype`
    - `size`
    - `shape`
    - `nbytes`
    """
    return {
        'type': type(arr),
        'dtype': arr.dtype,
        'size': arr.size,
        'shape': arr.shape,
        'nbytes': arr.data.nbytes
    }


def np_ncols(
        arr: np.ndarray) \
        -> Union[int, None]:
    """
    Number of columns.
    - 1d arrays are row vectors.
    - 0d arrays have no dimensions.
    """
    if arr.ndim == 0:
        return None
    elif arr.ndim == 1:
        return arr.shape[0]
    else:
        return arr.shape[1]


def np_nrows(
        arr: np.ndarray) \
        -> Union[int, None]:
    """
    Number of rows.
    - 1d arrays are row vectors.
    - 0d arrays have no dimensions.
    """
    if arr.ndim == 0:
        return None
    elif arr.ndim == 1:
        return 1
    else:
        return arr.shape[0]


def np_vstack(
        *arrs: Sequence) \
        -> np.ndarray:
    """
    `np.vstack` wrapper.
    """
    return np.vstack([*arrs])


def pairtable(
        arr: Sequence,
        dtype: str = 'object') \
        -> np.ndarray:
    """
    Upper triangular array.
    """
    n = triangle_n_inv(len(arr))
    table = np.empty((n, n), dtype=dtype)
    r_idx, c_idx = np.triu_indices(n)
    coords = np_vstack(r_idx, c_idx).T
    for i, (j, k) in enumerate(coords):
        table[j, k] = arr[i]
    return table


def seq(
        start,
        stop,
        step=None) \
        -> np.ndarray:
    """
    Inclusive sequence.
    """
    if step is None:
        if start < stop:
            step = 1
        else:
            step = -1

    if is_int(start) and is_int(step):
        dtype = 'int'
    else:
        dtype = None

    d = max(n_dec(step), n_dec(start))
    n_step = np.floor(round(stop - start, d + 1) / step) + 1
    delta = np.arange(n_step) * step
    return np.round(start + delta, decimals=d).astype(dtype)


def triangle_n(
        n: int) \
        -> int:
    """
    `n`th triangle number.

    See: https://en.wikipedia.org/wiki/Triangular_number
    """
    return round(n * (n + 1) / 2)


def triangle_n_inv(
        t: int) \
        -> Union[int, None]:
    """
    `n` given `n`th triangle number `t`.
    - Invalid `t` returns `None`.
    - Used to generate pair table.
    """
    coeff = (2 * t, -1, -1)
    root = np.max(polyroots(coeff))
    n = int(round(root))
    if triangle_n(n) == t:
        return n
    else:
        return None

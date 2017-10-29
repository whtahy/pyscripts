# Print utils
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

import numpy
from pyscripts.misc import max_length
from pyscripts.numbers import cartesian, seq, triangle_num_inv

if TYPE_CHECKING:
    from typing import *
    from pyscripts.types import *


# TODO: print_dict


def printf(
        string: 'Any',
        pad_char: str = ' ',
        l_padding: int = 0) \
        -> None:
    print(string, end = pad_char * l_padding)


def print_arr(
        arr: 'ndarray',
        padding: int = 2) \
        -> None:
    n_rows = arr.shape[0]
    n_cols = arr.T.shape[0]
    col_widths = numpy.zeros(n_cols, dtype = 'int')
    for c in range(n_cols):
        col_widths[c] = max_length(arr[:, c]) + padding * (c != 0)
    for r, c in cartesian(seq(0, n_rows, incl = False),
                          seq(0, n_cols, incl = False)):
        if arr[r, c] is None:
            val = '.'
        else:
            val = arr[r, c]
        printf(f'{val:>{col_widths[c]}}')
        if c == n_cols - 1:
            print()


def print_pairtable(
        arr: 'LT_Type',
        names: 'LT_Type' = None) \
        -> None:
    n_rc = triangle_num_inv(len(arr))
    table = numpy.empty((n_rc, n_rc), dtype = 'object')
    r, c = numpy.triu_indices(n_rc)
    coords = numpy.vstack((r, c)).T
    for i, (j, k) in enumerate(coords):
        table[j, k] = arr[i]
    print_table(table, names, names)


def print_table(
        data: 'ndarray',
        row_names: 'LT_Type' = None,
        col_names: 'LT_Type' = None) \
        -> None:
    if col_names is None:
        col_names = numpy.array([f'col_{i}' for i in range(data.T.shape[0])])
    if row_names is None:
        row_names = numpy.array([f'row_{i}' for i in range(data.shape[0])])

    left = numpy.vstack(('', row_names.reshape(-1, 1)))
    right = numpy.vstack((col_names, data))
    table = numpy.hstack((left, right))
    print_arr(table)

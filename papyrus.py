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
        printf(f'{arr[r,c]:>{col_widths[c]}}')
        if c == n_cols - 1:
            print()

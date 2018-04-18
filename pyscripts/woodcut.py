# Print utils
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from typing import Sequence

import numpy as np


def print_arr(
        arr: np.ndarray,
        sep: str=' '*2,
        align: str=None) \
        -> None:
    # reshape row -> col
    if arr.ndim < 2:
        arr = arr.reshape(-1, 1)

    # align and print floats
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.apply_along_axis(split_align, 0, arr)
        print(''.join([sep.join(row.astype('U')) + '\n' for row in arr]))
        return

    # calculate alignment
    if align:
        pass
    elif np.issubdtype(arr.dtype, np.integer):
        align = '>'
    else:
        align = '<'

    # calculate col widths
    col_width = []
    for col in arr.T:
        col_width += [max_len(col)]

    # print table
    s = []
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            s += [f'{arr[r,c]:{align}{col_width[c]}}' + sep]
        s += ['\n']
    print(''.join(s))


#
# Helpers


def max_len(
        row: Sequence) \
        -> int:
    """
    Max `len` of 1d array.
    """
    return max(len(str(x)) for x in row)


def split_align(
        row: Sequence,
        sep: str='.') \
        -> np.ndarray:
    """
    Align 1d array at `sep`.
    """
    split = np.char.partition(row.astype('U'), sep)

    left = split[:, 0]
    mid = split[:, 1]
    right = split[:, 2]

    w_left = max_len(left)
    w_right = max_len(right)

    align = np.char.add(
        np.char.rjust(left, w_left),
        mid
    )
    align = np.char.add(
        align,
        np.char.ljust(right, w_right)
    )

    return align

# Print utils
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from functools import partial
from typing import TYPE_CHECKING

import numpy
from numpy import max, sum

from pyscripts.commune import TERM_WIDTH
from pyscripts.zfc import (
    is_NLT, islastcol, numpy_hstack, numpy_ncols,
    numpy_vstack, pairtable, stralign_arr, strsplit_arr, type_map
)

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *


def print_arr(
        arr: 'ndarray',
        col_aligns: 'Union[NLT_StrType, str]' = None,
        row_names: 'NLT_Type' = None,
        col_names: 'NLT_Type' = None,
        n_decimals: int = -1,
        l_padding: int = 2,
        default_align: str = '>',
        max_rows: int = 100,
        warn: bool = True,
        max_width: int = TERM_WIDTH,
        replace_None_with: str = '.') -> None:
    # init
    full_nrows = arr.shape[0]
    full_ncols = numpy_ncols(arr)
    arr = arr.reshape(full_nrows, full_ncols).astype('object')

    # no. rows/cols
    n_rows = arr.shape[0]
    n_cols = numpy_ncols(arr)

    # n_decimals
    if n_decimals > 0:
        n_decimals += 1

    # stack row names
    if row_names is not None:
        arr = numpy_hstack(row_names.reshape(-1, 1), arr)
        n_cols = numpy_ncols(arr)

    # cut rows
    if n_rows > max_rows:
        arr = arr[0:max_rows]
        n_rows = max_rows

    # cut cols, prelim
    if n_cols > max_width:
        arr = arr[:, 0:max_width]
        n_cols = max_width

    # get col alignments using last row
    floatcode = 'f'
    p_typemap = partial(type_map,
                        type_names = ['float'],
                        codomain = [floatcode],
                        default_out = default_align)
    col_codes = numpy.vectorize(p_typemap)(arr[-1, :])  # last row

    # get col widths + align floats
    col_widths = numpy.empty(n_cols, dtype = 'uint')
    for i in numpy.arange(n_cols)[col_codes == floatcode]:
        # float col widths + align floats
        col_i = arr[:, i]
        col_i = numpy.where(col_i == None, '', col_i).astype('U')  # need ==
        left, right = strsplit_arr(col_i, by = '.')
        strarr, width = stralign_arr(left, right, maxlen_right = n_decimals)
        arr[:, i] = strarr
        col_widths[i] = width
    for i in numpy.arange(n_cols)[col_codes != floatcode]:
        # other col widths
        col_i = arr[:, i]
        col_i = numpy.where(col_i == None, '', col_i).astype('U')  # need ==
        col_widths[i] = max(numpy.vectorize(len)(col_i))

    # stack col names
    if col_names is not None:
        # cut col names
        col_names = col_names[0:n_cols]

        # check for row names
        if len(col_names) < n_cols:
            col_names = numpy_hstack([''], col_names)

        # vstack
        arr = numpy_vstack(col_names, arr)

        # update col widths & n_rows
        name_widths = numpy.vectorize(lambda x: len(str(x)))(col_names)
        col_widths = max(numpy_vstack(name_widths, col_widths), axis = 0)
        n_rows = arr.shape[0]

    # cut cols
    table_width = sum(col_widths) + l_padding * (n_cols - 1)
    if table_width > max_width:
        cum_widths = numpy.cumsum(col_widths) + l_padding * numpy.arange(n_cols)
        col_cutoff = numpy.searchsorted(cum_widths, max_width)
        arr = arr[:, 0:col_cutoff]
        table_width = cum_widths[0:col_cutoff][-1]
        n_cols = numpy_ncols(arr)
        col_widths = col_widths[0:n_cols]

    # col alignments
    if col_aligns:
        if not is_NLT(col_aligns):
            col_aligns = col_aligns.split()
    else:
        col_aligns = col_codes
        col_aligns[col_codes == floatcode] = default_align
        if row_names is not None:
            col_codes[0] = '<'

    # cutoff warning
    if warn and (n_cols < full_ncols or n_rows < full_nrows):
        printf(f'Showing {n_cols} / {full_ncols} columns')
        printf(f' ')
        printf(f'and {n_rows} / {full_nrows} rows.')
        print()
        print()

    # make table
    out = ''
    p_islastcol = partial(islastcol, arr = arr)
    for (r, c), item in numpy.ndenumerate(arr):
        if item is None:
            printval = replace_None_with
        else:
            printval = item
        out += f'{printval:{col_aligns[c]}{col_widths[c]}}'
        out += ' ' * l_padding * (not p_islastcol(c))
        if p_islastcol(c):
            out += '\n'
    print(out)


def print_dict(
        dictionary: dict,
        row_names: 'NLT_StrType' = None,
        col_names: 'NLT_StrType' = None,
        col_aligns: 'NLT_StrType' = None) \
        -> None:
    if col_aligns is None:
        col_aligns = ['<', '>']
    keys = list(dictionary.keys())
    vals = list(dictionary.values())
    arr = numpy_vstack(keys, vals).T
    print_arr(arr,
              col_aligns = col_aligns,
              row_names = row_names,
              col_names = col_names)


def print_each(
        nlt: 'NLT_Type') \
        -> None:
    for x in nlt:
        print(x)


def printf(
        string: 'Any',
        pad_with: str = ' ',
        n_padding: int = 0) \
        -> None:
    print(string, end = pad_with * n_padding)


def print_NLT(
        lst: 'NLT_Type') \
        -> None:
    for item in lst:
        print(item)


def print_pairtable(
        arr: 'NLT_Type',
        names: 'NLT_StrType' = None,
        col_aligns: 'NLT_StrType' = None) \
        -> None:
    print_arr(pairtable(arr), col_aligns, names, names)


def print_raw(
        arr: 'NLT_Type') \
        -> None:
    arr = numpy.array(arr, dtype = 'U')
    for row in arr:
        print(''.join(row))


def print_table(
        arr: 'ndarray',
        col_aligns: 'NLT_StrType' = None) \
        -> None:
    n_rows = arr.shape[0]
    n_cols = numpy_ncols(arr)
    arr = arr.reshape(n_rows, n_cols)
    row_names = numpy.arange(n_rows)
    col_names = numpy.arange(n_cols)
    print_arr(arr, col_aligns, row_names, col_names)

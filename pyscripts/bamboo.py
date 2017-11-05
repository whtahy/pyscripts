# Pandas utils
# Released under CC0:
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from typing import TYPE_CHECKING

import numpy
import pandas

from pyscripts.woodcut import printf

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *


# TODO
# Use woodcut


def count_dtypes(
        df: pandas.DataFrame) \
        -> 'pandas.Series':
    return df.dtypes.value_counts(ascending = True)


def features_by_dtype(
        df: 'pandas.DataFrame',
        print_out: bool = False,
        max_per_row: int = 10,
        min_rows: int = 1,
        col_padding: int = 2,
        terminal_width: int = 150) \
        -> dict:
    features_dict = dict()
    dtype_counts = count_dtypes(df)
    if print_out:
        print(f'{sum(dtype_counts)} features')
        print(f'{len(df.index)} rows,'
              f' '
              f'{df.isnull().values.sum().sum()} missing'
              f' '
              f'values')
        print()
    for dtype in [x for x in dtype_counts.index]:
        features = df.select_dtypes(include = [dtype]).columns.values
        features_dict[dtype.name] = list(features)
        if print_out:
            idx = idx_by_feature(df)
            length_index = len(str(len(features)))
            length_cols = max([len(str(x)) for x in features]) + col_padding
            names_per_row = min(terminal_width // (length_cols + length_index),
                                max_per_row)
            print(f'{dtype.name}: {dtype_counts[dtype]}')
            col_index = 0
            for feature in features:
                col_index += 1
                printf(f'{idx[feature]:>{length_index}}'
                       f' '
                       f'{feature:<{length_cols}}')
                if col_index % names_per_row == 0 \
                        or len(features) < min_rows * names_per_row \
                        or col_index >= len(features):
                    print()
            print()
    return features_dict


def idx_by_feature(
        df: pandas.DataFrame) \
        -> dict:
    return dict(zip(df.columns, numpy.arange(len(df.columns))))


def peek(
        file_path: str,
        usecols: 'LT_IntType' = None,
        show_dtypes: bool = False,
        show_preview: bool = False,
        nrows: int = 100,
        float_bits: int = 32,
        n_preview: int = 3,
        header = 'infer') \
        -> 'pandas.DataFrame':
    df = pandas.read_csv(file_path,
                         header = header,
                         nrows = nrows,
                         usecols = usecols)

    # Convert to float32
    for col in df.select_dtypes(include = ['float']):
        df[col] = df[col].astype(f'float{float_bits}')

    features_by_dtype(df, print_out = show_dtypes)
    if show_preview:
        print_preview(df, nrows = n_preview)
    return df


def print_preview(
        df: 'pandas.DataFrame',
        nrows: int = 3,
        cols: 'LT_IntType' = None) \
        -> None:
    dtype_counts = count_dtypes(df)
    print(f'{nrows} first rows')
    print()
    for dtype in dtype_counts.index:
        print(f'{dtype.name}: {dtype_counts[dtype]}')
        print(df.select_dtypes(include = [dtype]).iloc[0:nrows, cols])
        print()
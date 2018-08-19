# Numpy extensions
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import numpy as np


def na(X):
    count = X.isna().sum()
    if len(X.shape) < 2:
        return count
    else:
        return count[lambda x: x > 0]


def perc(x, n_dec=1, dtype='float'):
    return (x*100).astype(dtype).round(n_dec)


def reorder(df, order=None):
    """
    Sort `df` columns by dtype and name.
    """
    def sort(df):
        return df.dtypes.reset_index().sort_values([0, 'index'])['index']
    if order is None:
        order = [np.floating, np.integer, 'category', 'object']
    names = [sort(df.select_dtypes(s)) for s in order]
    return df[[x for ls in names for x in ls]]

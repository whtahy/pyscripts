# Transformers
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import numpy as np

import pandas as pd
from sklearn.base import TransformerMixin


class DataFrameUnion(TransformerMixin):
    def __init__(self, trf_list):
        self.trf_list = trf_list

    def fit(self, X, y=None):
        for t in self.trf_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        return pd.concat([t.transform(X) for t in self.trf_list], axis=1)


class FillNA(TransformerMixin):
    def __init__(self, col, val):
        self.col = col
        self.val = val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.col].fillna(self.val, inplace=True)
        return X


class GetDummies(TransformerMixin):
    def __init__(self, cols=None, drop_first=False):
        self.cols = cols
        self.drop = drop_first

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.cols:
            X = X[self.cols]
        return pd.get_dummies(X, drop_first=self.drop)


class NADummies(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(self.cols, str):
            return self.dummies(X, self.cols)
        else:
            ls = [self.dummies(X, c) for c in self.cols]
            return pd.concat(ls, axis=1)

    def dummies(self, X, col):
        return X[col].isna().astype(np.uint8).rename(col + '_na')


class PdStandardScaler(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.mean = X.mean()
        self.std = X.std(ddof=0)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


class PdTransform(TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.fn(X)


class SelectColumns(TransformerMixin):
    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.include:
            X = X[self.include]
        if self.exclude:
            return X.drop(columns=self.exclude)
        return X


class SelectDtypes(TransformerMixin):
    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(include=self.include, exclude=self.exclude)

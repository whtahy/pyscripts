# Transformers
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from typing import Iterable, Sequence, Tuple

import numpy as np

import pandas as pd
from pyscripts.zfc import np_hstack, np_ncols
from scipy.sparse import coo_matrix as coo
from scipy.sparse import csc_matrix as csc
from scipy.sparse import csr_matrix as csr
from sklearn.base import BaseEstimator, TransformerMixin, clone

# TODO: Switch to pd
# TODO: cleanup / refactor
# TODO: docs
# TODO: tests


#
# Transformers

class ModelTransformer(TransformerMixin):
    """
    Copyright 2014 Zac Stewart, MIT License
    """

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))


# TODO: Test with unseen values. Fix if broke.
class GetDummies(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.dtype = X.dtype
        return self

    def transform(self, X):
        return pd.get_dummies(X)

    def inverse_transform(self, X):
        return X.idxmax(axis=1).astype(self.dtype)


class TupleUnion(TransformerMixin, BaseEstimator):
    def __init__(self, trf_list):
        self.trf_list = trf_list

    def fit(self, X, y=None):
        for trf in self.trf_list:
            trf.fit(X, y)
        return self

    def transform(self, X):
        return tuple(trf.transform(X) for trf in self.trf_list)

    def predict(self, X):
        return tuple(trf.predict(X) for trf in self.trf_list)


#
# Old


class DictEncode(TransformerMixin, BaseEstimator):
    def __init__(self, encoding_dict: dict):
        self.encoding_dict = encoding_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return np.vectorize(self.encoding_dict.get)(X)


class Exp(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return np.exp(X)

    def inverse_transform(self, X) -> np.ndarray:
        return np.log(X)


class GetCols(TransformerMixin, BaseEstimator):
    def __init__(self, i_cols: Iterable[int] = None):
        self.cols = i_cols

    # Use pd to preserve dtypes!
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cols = X.iloc[:, self.cols].values
        if len(cols.shape) > 1:
            return cols
        else:
            return cols.reshape(-1, 1)


class InvTransform(TransformerMixin, BaseEstimator):
    def __init__(self, pips):
        self.pips = pips

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> Tuple[np.ndarray, np.ndarray]:
        left = X[0]
        right = X[1]
        inv_left = inv_cols(left, self.pips)
        inv_right = inv_cols(right, self.pips)
        return inv_left, inv_right


class Log(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return np.log(X)

    def inverse_transform(self, X) -> np.ndarray:
        return np.exp(X)


class OneHot(TransformerMixin, BaseEstimator):
    def __init__(self, dtype: str = bool, sparse_format: str = coo):
        self.dtype = dtype
        self.sparse_format = sparse_format
        self.X_min = None
        self.ncats_per_col = None
        self.col_widths = None

    def fit(self, X, y=None):
        self.X_min = np.min(X)
        self.ncats_per_col = np.max(X, axis=0) + (1 - self.X_min)
        cumsum = np.cumsum(self.ncats_per_col)
        self.col_widths = np.append([0], cumsum)[:-1]
        return self

    def transform(self, X) -> Tuple[coo, csc, csr]:
        row = np.arange(X.shape[0])
        row_idx = np.tile(row, (X.T.shape[0], 1)).T.ravel()
        col_idx = (self.col_widths + X - self.X_min).ravel()
        vals = np.ones(X.size, dtype=self.dtype)
        shape = (row.shape[0], np.sum(self.ncats_per_col))
        if self.sparse_format == csc:
            return csc((vals, (row_idx, col_idx)), shape=shape)
        elif self.sparse_format == csr:
            return csr((vals, (row_idx, col_idx)), shape=shape)
        elif self.sparse_format == coo:
            return coo((vals, (row_idx, col_idx)), shape=shape)
        else:
            return coo((vals, (row_idx, col_idx)), shape=shape).toarray()


class Power(TransformerMixin, BaseEstimator):
    def __init__(self, power: float):
        self.power = power

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return np.float_power(X, self.power)

    def inverse_transform(self, X) -> np.ndarray:
        return np.float_power(X, 1 / self.power)


class RavelCol(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return X.ravel()

    def inverse_transform(self, X) -> np.ndarray:
        return X.reshape(-1, 1)


class ReshapeAsCol(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return X.reshape(-1, 1)

    def inverse_transform(self, X) -> np.ndarray:
        return X.ravel()


class Restack(TransformerMixin, BaseEstimator):
    def __init__(self, n_models):
        self.n_models = n_models

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> Tuple[np.ndarray, np.ndarray]:
        left = restack(X[0], self.n_models)
        right = restack(X[1], self.n_models)
        return left, right


class Shift(TransformerMixin, BaseEstimator):
    def __init__(self, shift: float = 0):
        self.shift = shift

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return X + self.shift

    def inverse_transform(self, X) -> np.ndarray:
        return X - self.shift


class ToSparse(TransformerMixin, BaseEstimator):
    def __init__(self, sparse_format: str = coo) -> None:
        self.sparse_format = sparse_format

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return eval(f'scipy.sparse.{self.sparse_format}_matrix(X)')


# TODO: add _test oof (mean, median, min, max, etc.)
class TrainOOF(TransformerMixin, BaseEstimator):
    def __init__(self, X, model, kfold):
        self.X = X
        self.model = clone(model)
        self.kfold = kfold

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> Tuple[np.ndarray, np.ndarray]:
        ys = X.reshape(-1, np_ncols(X))
        oof = np.zeros(ys.shape)
        for i, y_i in enumerate(ys.T):
            for j, (train_idx, cv_idx) in enumerate(self.kfold.split(X)):
                _X = self.X[train_idx]
                _y = y_i[train_idx]
                _cv = self.X[cv_idx]

                self.model.fit(_X, _y)
                _hat = self.model.predict(_cv)
                oof[cv_idx, i] = _hat
        return oof, oof


#
# Helpers

def inv_cols(
        arr: np.ndarray,
        pips: Sequence) \
        -> np.ndarray:
    out = np.empty(arr.shape)
    for i, col in enumerate(arr.T):
        inv = pips[i].inverse_transform
        out[:, i] = inv(col.reshape(-1, 1))
    return out


def restack(
        arr: np.ndarray,
        n_cols: int) \
        -> np.ndarray:
    n_rows = int(len(arr) / n_cols)

    # [1, 2, 3, ..., n]
    # return np_vstack(
    #         *[arr[n_cols * i: n_cols * (i + 1)].reshape(1, -1)
    #           for i in range(n_rows)])

    # [1, n+1, 2n+1, 3n+1, ...]
    return np_hstack(
        *[arr[n_rows * i: n_rows * (i + 1)] for i in range(n_cols)])

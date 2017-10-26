# scikit-learn pipelines
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from typing import Iterable
    from numpy import ndarray


class DictEncode(TransformerMixin, BaseEstimator):
    def __init__(self, encode_dict: dict):
        self.encode_dict = encode_dict

    def fit(self, X, y = None):
        return self

    def transform(self, X) -> 'ndarray':
        # https://stackoverflow.com/a/16992783
        return numpy.vectorize(self.encode_dict.get)(X)


class Exp(TransformerMixin, BaseEstimator):
    def fit(self, X, y = None):
        return self

    def transform(self, X) -> 'ndarray':
        return numpy.exp(X)

    def inverse_transform(self, X) -> 'ndarray':
        return numpy.log(X)


class GetCols(TransformerMixin, BaseEstimator):
    def __init__(self, i_cols: 'Iterable[int]' = None):
        self.cols = i_cols

    # Use pandas to preserve dtypes!
    def fit(self, X: 'pandas.DataFrame', y = None):
        return self

    def transform(self, X: 'pandas.DataFrame') -> 'ndarray':
        cols = X.iloc[:, self.cols].values
        if len(cols.shape) > 1:
            return cols
        else:
            return cols.reshape(-1, 1)


class Log(TransformerMixin, BaseEstimator):
    def fit(self, X, y = None):
        return self

    def transform(self, X) -> 'ndarray':
        return numpy.log(X)

    def inverse_transform(self, X) -> 'ndarray':
        return numpy.exp(X)


class OneHot(TransformerMixin, BaseEstimator):
    def __init__(self, dtype: str = 'bool', sparse_format: str = 'coo'):
        self.dtype = dtype
        self.sparse_format = sparse_format
        self.X_min = None
        self.ncats_per_col = None
        self.col_widths = None

    def fit(self, X, y = None):
        self.X_min = X.min()
        self.ncats_per_col = numpy.max(X, axis = 0) + (1 - self.X_min)
        self.col_widths = numpy.append([0],
                                       numpy.cumsum(self.ncats_per_col))[:-1]
        return self

    def transform(self, X) -> 'ndarray':
        row = numpy.arange(X.shape[0])
        row_idx = numpy.tile(row, (X.shape[1], 1)).T.ravel()
        col_idx = (self.col_widths + X - self.X_min).ravel()
        vals = numpy.ones(X.size, dtype = self.dtype)
        shape = (row.shape[0], numpy.sum(self.ncats_per_col))
        arg = f'({vals}, ({row_idx}, {col_idx})), shape = {shape}'
        return eval(f'scipy.sparse.{self.sparse_format}_matrix({arg})')


class Power(TransformerMixin, BaseEstimator):
    def __init__(self, power: float):
        self.power = power

    def fit(self, X, y = None):
        return self

    def transform(self, X) -> 'ndarray':
        return numpy.float_power(X, self.power)

    def inverse_transform(self, X) -> 'ndarray':
        return numpy.float_power(X, 1 / self.power)


class RavelCol(TransformerMixin, BaseEstimator):
    def fit(self, X, y = None):
        return self

    def transform(self, X) -> 'ndarray':
        return X.ravel()

    def inverse_transform(self, X) -> 'ndarray':
        return X.reshape(-1, 1)


class ReshapeAsCol(TransformerMixin, BaseEstimator):
    def fit(self, X, y = None):
        return self

    def transform(self, X) -> 'ndarray':
        return X.reshape(-1, 1)

    def inverse_transform(self, X) -> 'ndarray':
        return X.ravel()


class Shift(TransformerMixin, BaseEstimator):
    def __init__(self, shift: float = 0):
        self.shift = shift

    def fit(self, X, y = None):
        return self

    def transform(self, X) -> 'ndarray':
        return X + self.shift

    def inverse_transform(self, X) -> 'ndarray':
        return X - self.shift


class ToSparse(TransformerMixin, BaseEstimator):
    def __init__(self, sparse_format: str = 'coo') -> None:
        self.sparse_format = sparse_format

    def fit(self, X, y = None):
        return self

    def transform(self, X) -> 'ndarray':
        return eval(f'scipy.sparse.{self.sparse_format}_matrix(X)')

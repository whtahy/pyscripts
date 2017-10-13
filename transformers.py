import pandas
import numpy
import math
import scipy

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator


class ReshapeAsCol(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(-1,1)

    def inverse_transform(self, X):
        return X.ravel()


class RavelCol(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.ravel()

    def inverse_transform(self, X):
        return X.reshape(-1,1)


class GetCols(TransformerMixin, BaseEstimator):
    '''
    Use pandas to preserve dtypes!
    '''
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = X.iloc[:, self.cols].values
        if len(cols.shape) > 1:
            return cols
        else:
            return cols.reshape(-1,1)


class DictEncode(TransformerMixin, BaseEstimator):
    def __init__(self, encode_dict):
        self.encode_dict = encode_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # https://stackoverflow.com/a/16992783
        return numpy.vectorize(self.encode_dict.get)(X)


class OneHot(TransformerMixin, BaseEstimator):
    def __init__(self, dtype='bool', sparse_format='coo'):
        self.dtype = dtype
        self.sparse_format = sparse_format

    def fit(self, X, y=None):
        self.X_min = X.min()
        self.ncats_per_col = numpy.max(X, axis=0) + (1 - self.X_min)
        self.col_widths = numpy.append(0, numpy.cumsum(self.ncats_per_col))[:-1]
        self.col = numpy.arange(X.shape[1])
        return self

    def transform(self, X):
        row = numpy.arange(X.shape[0])
        row_idx = numpy.tile(row, (X.shape[1],1)).T.ravel()
        col_idx = (self.col_widths + X - self.X_min).ravel()
        vals = numpy.ones(X.size, dtype=self.dtype)
        shape = (row.shape[0], numpy.sum(self.ncats_per_col))
        arg = '(vals, (row_idx, col_idx)), shape=shape'
        return eval(f'scipy.sparse.{self.sparse_format}_matrix({arg})')


class ToSparse(TransformerMixin, BaseEstimator):
    def __init__(self, sparse_format='coo'):
        self.sparse_format = sparse_format

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return eval(f'scipy.sparse.{self.sparse_format}_matrix(X)')
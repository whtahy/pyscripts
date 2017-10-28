# Mypy static type aliases
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *
    from numpy import ndarray
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

SparseType = Union[coo_matrix, csc_matrix, csr_matrix]
FlexIntType = Union[int, Iterable[int]]
FlexFloatType = Union[float, Iterable[float]]
NpIntType = Union[FlexIntType, ndarray]
NpFloatType = Union[FlexFloatType, ndarray]
NpStrType = Union[str, ndarray]

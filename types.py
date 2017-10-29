# Mypy static type aliases
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *
    from numpy import ndarray
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

FlexFloatType = Union[float, List[float], Tuple[float], ndarray]
FlexIntType = Union[int, List[int], Tuple[int], ndarray]
FlexStrType = Union[str, List[str], Tuple[str], ndarray]
LT_FloatType = Union[List[float], Tuple[float], ndarray]
LT_IntType = Union[List[int], Tuple[int], ndarray]
LT_Type = Union(List, Tuple, ndarray)
SparseType = Union[coo_matrix, csc_matrix, csr_matrix]

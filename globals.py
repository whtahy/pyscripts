# Global constants
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

SEED: int = 123

# 1680 x 1050 resolution
PYPLOT_WIDTH: int = 26
PYPLOT_HEIGHT: int = 17


SparseType = 'Union[coo_matrix, csc_matrix, csr_matrix]'
FlexIntType = 'Union[int, Iterable[int]]'
FlexFloatType = 'Union[float, Iterable[float]]'
NpIntType = 'Union[FlexIntType, ndarray]'
NpFloatType = 'Union[FlexFloatType, ndarray]'
NpStrType = 'Union[str, ndarray]'

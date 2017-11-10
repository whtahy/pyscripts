# Mypy static type aliases
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *
    from numpy import ndarray
    from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
    from sklearn.pipeline import Pipeline, FeatureUnion

NLT_Type = Union[ndarray, List, Tuple]
NLT_FloatType = Union[ndarray, List[float], Tuple[float]]
NLT_IntType = Union[ndarray, List[int], Tuple[int]]
NLT_StrType = Union[ndarray, List[str], Tuple[str]]

FlexFloatType = Union[float, NLT_FloatType]
FlexIntType = Union[int, NLT_IntType]
FlexStrType = Union[str, NLT_StrType]
FlexType = Union[FlexFloatType, FlexIntType, FlexStrType]

SparseType = Union[csc_matrix, csr_matrix, coo_matrix]

Pip = Pipeline
Union = FeatureUnion

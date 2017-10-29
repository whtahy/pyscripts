# Misc utils
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


import itertools
from typing import TYPE_CHECKING

import numpy
from scipy.misc import comb

if TYPE_CHECKING:
    from typing import *
    from pyscripts.types import *


def max_length(
        arr: 'LT_Type') \
        -> int:
    return max([len(str(x)) for x in arr])


def subset_apply(
        arr: 'LT_Type',
        func: 'Callable',
        l_subset: int = 2) \
        -> 'ndarray':
    out = numpy.empty(comb(len(arr), l_subset).astype('int'),
                      dtype = 'object')
    for i, subset in enumerate(itertools.combinations(arr, l_subset)):
        out[i] = func(*subset)
    return out


def subdict(
        full_dict: dict,
        keys: 'LT_Type') \
        -> dict:
    return {k: v for k, v in full_dict.items() if k in keys}

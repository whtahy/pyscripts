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

    T = TypeVar('T')


def max_length(
        arr: 'Iterable') \
        -> int:
    return max([len(str(x)) for x in arr])


def subset_apply(
        objects: 'Iterable',
        func: 'Callable',
        l_subset: int = 2) \
        -> 'ndarray':
    out = numpy.empty(comb(len(objects), l_subset).astype('int'),
                      dtype = 'object')
    for i, subset in enumerate(itertools.combinations(objects, l_subset)):
        out[i] = func(*subset)
    return out


def subdict(
        full_dict: 'Dict[T]',
        keys: 'Iterable[Any]') \
        -> 'Dict[T]':
    return {k: v for k, v in full_dict.items() if k in keys}

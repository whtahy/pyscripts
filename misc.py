# Misc utils
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *
    from pyscripts.types import *

    T = TypeVar('T')


def subset_apply(
        objects: 'Iterable[Any]',
        func: 'Callable[..., T]',
        l_subset: int = 2) \
        -> 'List[T]':
    out = []
    for subset in itertools.combinations(objects, l_subset):
        out += [func(*subset)]
    return out


def subdict(
        full_dict: 'Dict[T]',
        keys: 'Iterable[Any]') \
        -> 'Dict[T]':
    return {k: v for k, v in full_dict.items() if k in keys}

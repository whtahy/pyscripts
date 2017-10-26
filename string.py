# Copy/paste
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Tuple, Union
    from numpy import ndarray


def char(
        offset: int,
        start: str = 'a') \
        -> str:
    return chr(ord(start) + offset)


def chars(
        offsets: 'Union[int, ndarray, List[int], Tuple[int]]',
        start: str = 'a') \
        -> 'Union[str, List[str]]':
    if type(offsets) is int or len(offsets) == 1:
        return char(start = start, offset = offsets)
    else:
        return list(map(partial(char, start = start), offsets))


def printf(
        s: str) \
        -> None:
    print(s, end = '')


def strip_ext(
        s: str,
        ext: str,
        sep: str = '.') \
        -> str:
    if ext[0] == sep:
        ext = ext[1:]
    if s.endswith(f'{sep}{ext}'):
        return s[:-len(ext) - len(sep)]
    else:
        return s

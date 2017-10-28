# String utils
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *
    from pyscripts.types import *


def char(
        offset: int,
        start: str = 'a') \
        -> str:
    return str_shift(start, offset)


def chars(
        offsets: 'FlexIntType',
        start: str = 'a') \
        -> 'ndarray':
    if type(offsets) is int or len(offsets) == 1:
        return numpy.array(char(start = start, offset = offsets))
    else:
        return numpy.fromiter(map(partial(char, start = start), offsets),
                              dtype = 'U1')


def codes_decimal() -> 'ndarray':
    return seq(ord('0'), ord('9'))


def codes_lower() -> 'ndarray':
    return seq(ord('a'), ord('z'))


def codes_upper() -> 'ndarray':
    return seq(ord('A'), ord('Z'))


def extract_alpha(
        string: str) \
        -> str:
    return str_filter(string, func = str.isalpha)


def extract_decimal(
        string: str) \
        -> str:
    return str_filter(string, func = str.isdecimal)


def str_apply(
        string: str,
        func: 'Callable[str, str]') \
        -> str:
    out = [func(s) for s in string]
    return ''.join(out)


def str_filter(
        string: str,
        func: 'Callable[str, bool]') \
        -> str:
    out = [s for s in string if func(s)]
    return ''.join(out)


def str_shift(
        string: str,
        shift: int) \
        -> str:
    return chr(ord(string) + shift)


def put_ext(
        string: str,
        ext: str,
        sep: str = '.') \
        -> str:
    return strip_ext(string, ext, sep) + sep + ext


def strip_ext(
        string: str,
        ext: str,
        sep: str = '.') \
        -> str:
    if ext[0] == sep:
        ext = ext[1:]
    if string.endswith(f'{sep}{ext}'):
        return string[:-len(ext) - len(sep)]
    else:
        return string

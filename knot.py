# String utils
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

import numpy
from pyscripts.numbers import seq

if TYPE_CHECKING:
    from typing import *
    from pyscripts.types import *


def chr_shift(
        chr_s: str,
        shift: int) \
        -> str:
    return chr(ord(chr_s) + shift)


def chr_shifts(
        chr_s: str,
        shifts: 'LT_IntType') \
        -> 'ndarray':
    return numpy.array([chr_shift(chr_s, sh) for sh in shifts])


def chr_codes_decimal() -> 'ndarray':
    return seq(ord('0'), ord('9'))


def chr_codes_lower() -> 'ndarray':
    return seq(ord('a'), ord('z'))


def chr_codes_upper() -> 'ndarray':
    return seq(ord('A'), ord('Z'))


def chrs(
        arr: 'LT_IntType') \
        -> 'ndarray':
    return numpy.array([chr(x) for x in arr])


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

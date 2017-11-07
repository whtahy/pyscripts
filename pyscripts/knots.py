# String utils
# Released under CC0:
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import os
from typing import TYPE_CHECKING

import numpy

from pyscripts.hero import re_ext, re_slash
from pyscripts.zfc import is_NLT, seq, weave

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *

CHR_CODES_DECIMAL = seq(ord('0'), ord('9'))
CHR_CODES_LOWER = seq(ord('a'), ord('z'))
CHR_CODES_UPPER = seq(ord('A'), ord('Z'))


def chr_apply(
        string: str,
        func: 'Callable[str, str]') \
        -> str:
    return ''.join([func(s) for s in string])


def chr_codes(
        string: str) \
        -> 'ndarray':
    return numpy.vectorize(ord)(list(string))


def chr_shift(
        chr_string: str,
        shift: int) \
        -> str:
    return chr(ord(chr_string) + shift)


def chrs(
        arr: 'NLT_IntType') \
        -> 'ndarray':
    return numpy.vectorize(chr)(arr).astype('object')


def extract_ext(
        string: str) \
        -> str:
    re_match = re_ext.search(string)
    if re_match:
        return re_match.group(1)[1:]
    else:
        return ''


def os_path(
        string: str) \
        -> str:
    pieces = path_pieces(string)
    n_sep = len(pieces) - 1
    return ''.join(weave(pieces, [os.sep] * n_sep))


def str_replace(
        string: str,
        replace_these: 'NLT_StrType',
        with_this: str = '') \
        -> str:
    out = string
    if is_NLT(replace_these):
        for s in replace_these:
            out = out.replace(s, with_this)
        return out
    else:
        return out.replace(string, with_this)


def strip_ext(
        string: str) \
        -> str:
    ext = extract_ext(string)
    return string.replace('.' + ext, '')


def path_pieces(
        path: str) \
        -> 'ndarray':
    return numpy.array(re_slash.split(path), dtype = 'object')

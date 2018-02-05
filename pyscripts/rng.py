# Random sampling
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from typing import TYPE_CHECKING

import numpy
import randomstate.prng.pcg64 as pcg
from numpy import sum

from pyscripts.knot import (
    CHR_CODES_DECIMAL,
    CHR_CODES_LOWER,
    CHR_CODES_UPPER,
    chrs
)
from pyscripts.zfc import overflow_check

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *


# TODO: r_onehot


def bootstrap(
        arr: 'NLT_Type',
        n_draws: int = None):
    if n_draws is None:
        n_draws = len(arr)
    return arr[r_ints(n_draws, 0, len(arr), incl = False)]


def permute(
        arr: 'NLT_Type',
        n_draws: int = None) \
        -> 'ndarray':
    n_rows = len(arr)
    if n_draws is None:
        n_draws = n_rows
    idx = pcg.permutation(n_rows)[0:n_draws]
    return arr[idx]


def flip(
        pr_success: float = 0.5,
        n_sigfig: int = 3) \
        -> 'bool':
    if pr_success < 0 or pr_success > 1:
        raise ValueError("pr_success should be in interval [0,1]")
    high = 10 ** n_sigfig
    return r_int(1, high) <= pr_success * high


# TODO: Add static type for return
def get_state():
    return pcg.get_state()


def roll(
        dice: str) \
        -> 'Tuple[int, ndarray]':
    pivot = dice.find('d')
    n_dice = int(dice[0: pivot])
    n_sides = int(dice[pivot + 1: len(dice)])
    rolls = r_ints(n_dice, 1, n_sides)
    out = sum(rolls)
    return out, rolls


def r_chr(
        code_pool: 'NLT_IntType' = CHR_CODES_UPPER) \
        -> str:
    return chr(r_draw(code_pool))


def r_chrs(
        n_chars: int = 10,
        code_pool: 'NLT_IntType' = CHR_CODES_UPPER) \
        -> 'ndarray':
    return chrs(bootstrap(code_pool, n_chars))


def r_chr_upper() -> str:
    return chr(r_draw(CHR_CODES_UPPER))


def r_chr_lower() -> str:
    return chr(r_draw(CHR_CODES_LOWER))


def r_chr_decimal() -> str:
    return chr(r_draw(CHR_CODES_DECIMAL))


def r_draw(
        arr: 'NLT_Type') \
        -> 'Any':
    return arr[r_index(arr)]


def r_index(
        arr: 'NLT_Type') \
        -> int:
    return r_int(0, len(arr), False)


def r_int(
        low: int = -10,
        high: int = 10,
        incl: bool = True) \
        -> int:
    return r_ints(1, low, high, incl)[0]


def r_ints(
        n_ints: 'int' = 10,
        low: int = -10,
        high: int = 10,
        incl: bool = True) \
        -> 'ndarray':
    dtype_name = overflow_check(low, high)
    return pcg.randint(low, high + incl, n_ints, dtype_name)


# TODO: vectorize
def r_string(
        length: int = None,
        str_format: str = None) \
        -> str:
    out = ''
    if str_format is None:
        if length is None:
            length = r_int(5, 10)
        buckets = [CHR_CODES_DECIMAL, CHR_CODES_LOWER, CHR_CODES_UPPER]
        for i in range(length):
            bucket = r_draw(buckets)
            out += chr(r_draw(bucket))
    else:
        for i, c in enumerate(str_format):
            code = ord(c)
            if code in CHR_CODES_DECIMAL:
                out += r_chr_decimal()
            elif code in CHR_CODES_LOWER:
                out += r_chr_lower()
            elif code in CHR_CODES_UPPER:
                out += r_chr_upper()
            else:
                out += c
    return out


# TODO: speedup
def r_strings(
        n_strings: int = 10,
        length: int = None,
        str_format: str = None) \
        -> 'ndarray':
    out = numpy.empty(n_strings, dtype = 'object')
    for i in range(n_strings):
        out[i] = r_string(length, str_format)
    return out


def seed(
        number: int) \
        -> None:
    pcg.seed(number)


# TODO: Add static type for arg
def set_state(
        state) \
        -> None:
    pcg.set_state(state)

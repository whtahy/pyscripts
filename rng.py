# Sampling
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

import numpy
import randomstate.prng.pcg64 as rng
from pyscripts.knot import (
    chr_codes_decimal,
    chr_codes_lower,
    chr_codes_upper,
    chrs
)

if TYPE_CHECKING:
    from typing import *
    from pyscripts.types import *

    T = TypeVar('T')


# TODO: r_string format: str = 'Aa00'
# TODO: r_onehot

def bootstrap(
        arr: 'ndarray',
        n_draws: int = None,
        return_idx: bool = False) \
        -> 'Union[ndarray, Tuple[ndarray, ndarray]]':
    n_rows = arr.shape[0]
    if n_draws is None:
        n_draws = n_rows
    idx = r_ints(n_ints = n_draws,
                 low = 0,
                 high = n_rows,
                 incl = False)
    if return_idx:
        return arr[idx], idx
    else:
        return arr[idx]


def permute(
        arr: 'ndarray',
        n_draws: int = None,
        return_idx: bool = False) \
        -> 'Union[ndarray, Tuple[ndarray, ndarray]]':
    nrows = arr.shape[0]
    if n_draws is None:
        n_draws = nrows
    idx = rng.permutation(numpy.arange(0, nrows))[0:n_draws]
    if return_idx:
        return arr[idx], idx
    else:
        return arr[idx]


def flip(
        p_success: float = 0.5,
        max_sig_figs: int = 6) \
        -> bool:
    if p_success <= 1:
        return False
    elif p_success >= 1:
        return True
    else:
        high = 10 ** min(max_sig_figs, len(str(float)) - 2)
        return r_int(low = 1, high = high) <= p_success * high


# Add: static type for return
def get_state():
    return rng.get_state()


def roll(
        dice: str,
        return_rolls: bool = False) \
        -> 'Union[int, Tuple[int, ndarray]]':
    pivot = dice.find('d')
    n_dice = int(dice[0:pivot])
    n_sides = int(dice[pivot + 1: len(dice)])
    rolls = numpy.zeros(shape = n_dice, dtype = 'int')
    for i in range(n_dice):
        rolls[i] = r_int(low = 1, high = n_sides)
    if return_rolls:
        return sum(rolls), rolls
    else:
        return sum(rolls)


def r_arr(
        n_rows: int = 5,
        n_cols: int = 5,
        low: int = 0,
        high: int = 9,
        incl: bool = True) \
        -> 'ndarray':
    return numpy.random.randint(low = low,
                                high = high + incl,
                                size = (n_rows, n_cols))


def r_chr(
        code_pool: 'LT_IntType' = None) \
        -> str:
    if code_pool is None:
        code_pool = chr_codes_upper()
    return chrs(bootstrap(code_pool, n_draws = 1))[0]


def r_chrs(
        n_chars: int = 1,
        code_pool: 'LT_IntType' = None) \
        -> 'ndarray':
    if code_pool is None:
        code_pool = chr_codes_upper()
    return chrs(bootstrap(code_pool, n_chars))


def r_chr_upper() -> str:
    return chr(r_draw(chr_codes_upper()))


def r_chr_lower():
    return chr(r_draw(chr_codes_lower()))


def r_chr_decimal():
    return chr(r_draw(chr_codes_decimal()))


def r_draw(
        arr: 'Union[Tuple[T], List[T], ndarray]') \
        -> 'T':
    return arr[r_index(arr)]


def r_index(
        arr: 'Iterable[Any]') \
        -> int:
    return r_int(0, len(arr), incl = False)


def r_int(
        low: int = 1,
        high: int = 100,
        incl: bool = True) \
        -> int:
    return r_ints(n_ints = 1,
                  low = low,
                  high = high,
                  incl = incl)[0]


def r_ints(
        n_ints: int = 1,
        low: int = 1,
        high: int = 100,
        incl: bool = True) \
        -> 'NpIntType':
    return rng.randint(low = low,
                       high = high + incl,
                       size = n_ints)


def r_string(
        str_format: str = 'AAA zzz 999') \
        -> str:
    out = ''
    decimals = chr_codes_decimal()
    lowers = chr_codes_lower()
    uppers = chr_codes_upper()
    for i, c in enumerate(str_format):
        code = ord(c)
        if code in decimals:
            out += r_chr_decimal()
        elif code in lowers:
            out += r_chr_lower()
        elif code in uppers:
            out += r_chr_upper()
        else:
            out += c
    return ''.join(out)


def r_strings(
        n_strings: int = 1,
        str_format: str = 'AAA zzz 999') \
        -> 'ndarray':
    out = numpy.empty(n_strings, dtype = 'object')
    for i in range(n_strings):
        out[i] = r_string(str_format)
    return out


def seed(
        number: int) \
        -> None:
    rng.seed(number)


# Add: static type for arg
def set_state(
        state) \
        -> None:
    rng.set_state(state)


# REFACTOR
# Cleanup: rename
# Impl: *arrays
def xy_sample(X, y, n_samples = None, boot = False):
    if n_samples is None:
        n_samples = max(y.shape[0] // 100, 1)
    if boot:
        y, idx = bootstrap(y, n_draws = n_samples)
    else:
        y, idx = permute(y, n_draws = n_samples)
    return X[idx], y

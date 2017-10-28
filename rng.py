# Sampling
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

import numpy
import randomstate.prng.pcg64 as rng
from pyscripts.knot import char, chars, codes_decimal, codes_lower, codes_upper

if TYPE_CHECKING:
    from typing import *
    from pyscripts.types import *


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


def r_char(
        start: str = 'a',
        high: int = 26) \
        -> str:
    return char(r_int(low = 0, high = high, incl = False),
                start = start)


def r_chars(
        n_chars: int = 1,
        start: str = 'a',
        high: int = 26) \
        -> 'ndarray':
    return chars(r_ints(n_chars, low = 0, high = high, incl = False),
                 start = start)


def r_char_upper():
    return r_char(min(codes_upper()), max(codes_upper()))


def r_char_lower():
    return r_char(min(codes_lower()), max(codes_lower()))


def r_char_decimal():
    return r_char(min(codes_decimal()), max(codes_decimal()))


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
        str_format: str = 'AAAzzz999') \
        -> str:
    out = ''
    decimals = codes_decimal()
    lowers = codes_lower()
    uppers = codes_upper()
    for i, c in enumerate(str_format):
        if c in decimals:
            out += r_char_decimal()
        elif c in lowers:
            out += r_char_lower()
        elif c in uppers:
            out += r_char_upper()
    return ''.join(out)


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

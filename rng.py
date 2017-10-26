#
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

import numpy
import randomstate.prng.pcg64 as rng
from pyscripts.string import chars

if TYPE_CHECKING:
    from typing import Union, Tuple, List
    from numpy import ndarray


# TODO: Generate alphanumeric data
# TODO: Generate integer data
# TODO: Generate onehot data

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
        high: int = 100) \
        -> bool:
    return r_int(low = 1, high = high) <= p_success * high


# Need return type
def get_state():
    return rng.get_state()


def r_chars(
        n_chars: int = 1,
        start: str = 'a') \
        -> 'Union[str, numpy.ndarray]':
    return chars(r_ints(n_chars, low = 0, high = 26, incl = False),
                 start = start)


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
        -> 'ndarray':
    return rng.randint(low = low,
                       high = high + incl,
                       size = n_ints)


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


def roll(
        dice: str,
        return_rolls: bool = False) \
        -> 'Union[int, Tuple[int, ndarray]]':
    pivot = dice.find('d')
    n_dice = int(dice[0:pivot])
    n_sides = int(dice[pivot + 1: len(dice)])
    rolls = numpy.zeros(shape = n_dice, dtype = 'int')
    for i in range(n_dice):
        rolls[i] = r_int(high = n_sides)
    if return_rolls:
        return sum(rolls), rolls
    else:
        return sum(rolls)


def seed(
        number: int) \
        -> None:
    rng.seed(number)


# get arg type
def set_state(
        state) \
        -> None:
    rng.set_state(state)


# REFACTOR
def xy_sample(X, y, n_samples = None, boot = False):
    if n_samples is None:
        n_samples = max(y.shape[0] // 100, 1)
    if boot:
        y, idx = bootstrap(y, n_draws = n_samples)
    else:
        y, idx = permute(y, n_draws = n_samples)
    return X[idx], y

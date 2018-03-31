# Random sampling
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import string
from typing import TYPE_CHECKING

from numpy import random as mt19937

from pyscripts.zfc import npa, overflow_check

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *

# TODO: r_onehot
# TODO: sane implementation

word_list = npa([
    'a',
    'an',
    'so',
    'the',
    'and',
    'dart',
    'that',
    'quart',
    'danger',
    'program',
    'exercise',
    'arguments',
    'regulation',
])


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
    idx = mt19937.permutation(n_rows)[0:n_draws]
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
    return mt19937.get_state()


def roll(
        dice: str) \
        -> 'Tuple[int, ndarray]':
    pivot = dice.find('d')
    n_dice = int(dice[0: pivot])
    n_sides = int(dice[pivot + 1: len(dice)])
    rolls = r_ints(n_dice, 1, n_sides)
    out = int(sum(rolls))
    return out, rolls


def r_char(
        pool: 'NLT_IntType' = npa(list(string.ascii_letters))) \
        -> str:
    return r_draw(pool)


def r_chars(
        n_chars: int = 10,
        pool: 'NLT_IntType' = npa(list(string.ascii_letters))) \
        -> 'ndarray':
    return bootstrap(pool, n_chars)


def r_char_upper() -> 'str':
    return r_draw(npa(list(string.ascii_uppercase)))


def r_char_lower() -> 'str':
    return r_draw(npa(list(string.ascii_lowercase)))


def r_char_digit() -> 'str':
    return r_draw(npa(list(string.digits)))


def r_draw(
        arr: 'NLT_Type') \
        -> 'Any':
    return arr[r_index(arr)]


def r_index(
        arr: 'NLT_Type') \
        -> int:
    return r_int(0, len(arr), incl = False)


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
    return mt19937.randint(low, high + incl, n_ints, dtype_name)


def r_string(
        str_length: int = 10) \
        -> str:
    pool = npa(list(string.ascii_letters + string.digits))
    return ''.join(bootstrap(pool, str_length))


def r_stringf(
        str_format: str = 'Aaaaa') \
        -> str:
    def f(ch):
        if ch in string.ascii_lowercase:
            return r_char_lower()
        elif ch in string.ascii_uppercase:
            return r_char_upper()
        elif ch in string.digits:
            return r_char_digit()

    if str_format:
        return ''.join(map(f, str_format))


def r_strings(
        n_strings: int = 10,
        str_length: int = 10) \
        -> 'ndarray':
    total = str_length * n_strings
    s = r_string(total)
    return npa([s[i: i + str_length] for i in range(0, total, str_length)])


def r_text(n_words = 100, word_list = word_list):
    return ' '.join(bootstrap(word_list, n_words))


def seed(
        number: int) \
        -> None:
    mt19937.seed(number)


# TODO: Add static type for arg
def set_state(
        state) \
        -> None:
    mt19937.set_state(state)

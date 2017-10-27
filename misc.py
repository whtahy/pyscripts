# Misc utils
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


import itertools
from typing import TYPE_CHECKING

import numpy
from numpy import absolute as abs, median
from pandas import read_csv
from pyscripts.knot import printf
from pyscripts.numbers import seq, pslice
from scipy.stats import percentileofscore as percentileof

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, List, Tuple


def subset_apply(
        *objects: 'Any',
        l_subset: int = 2,
        func: 'Callable') \
        -> 'List[Any]':
    out = []
    for subset in itertools.combinations(objects, l_subset):
        out += [func(*subset)]
    return out


def leaderboard(
        score: float = None,
        file_name: str = 'leaderboard.csv',
        col_name: str = 'Score',
        label_left = 'Percentile',
        label_right = 'Score',
        higher_is_better = True,
        step = 10,
        padding = 3,
        decimals = 0) \
        -> 'Tuple':
    if higher_is_better:
        percents = seq(100, step, -step)
    else:
        percents = seq(0, 100 - step, step)
    scores = read_csv(file_name, usecols = [col_name]).values
    med_score = median(scores)
    width_left = len(label_left)
    width_right = max(len(str(med_score)), len(label_right)) + padding
    if score is not None:
        percentile_rank = abs(
                100 * (not higher_is_better) - percentileof(scores, score))
        printf(f'{"Rank":>{width_left}}')
        printf(f'{percentile_rank:>{width_right-1}.{decimals}f}%')
        print('\n')
    printf(f'{label_left:>{width_left}}')
    printf(f'{label_right:>{width_right}}')
    print()
    scores_pslice = pslice(scores, percents)
    ptiles = seq(100, 0, step = -step, incl = False)
    for i in range(len(scores_pslice)):
        printf(f'{ptiles[i]:>{width_left}}')
        printf(f'{scores_pslice[i]:>{width_right}.{decimals}f}')
        print()
    return ptiles, scores_pslice


def subdict(full_dict: dict,
            keys: 'Iterable[Any]'):
    return {k: v for k, v in full_dict.items() if k in keys}

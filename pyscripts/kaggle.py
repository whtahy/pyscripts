# Kaggle utils
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from typing import TYPE_CHECKING

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from scipy.stats import percentileofscore

from pyscripts.commune import PYPLOT_HEIGHT, PYPLOT_WIDTH
from pyscripts.woodcut import printf
from pyscripts.zfc import pslice_top, seq

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *


# TODO: use print_table
def leaderboard(
        percents: 'NLT_FloatType' = None,
        score: int = None,
        n_decimals: int = 0,
        higher_is_better: bool = True,
        show_rank: bool = True,
        show_ptile: bool = True,
        show_pslice: bool = True,
        show_plot: bool = True,
        multiplier: int = 1,
        plot_percents: int = 50) \
        -> None:
    if score is None:
        show_ptile = False
        show_rank = False
    scores = read_leaderboard() * multiplier
    if percents is not None:
        percents = numpy.sort(percents)[::-1]
    if show_ptile:
        ptile = leaderboard_ptile(score, scores, higher_is_better)
        print(f'Percentile: {ptile:.1f}%')
    if show_rank:
        rank = leaderboard_rank(score, scores, higher_is_better)
        print(f'Rank: {rank} of {len(scores)}')
    if show_pslice:
        if show_rank or show_ptile:
            print()
        x, y = leaderboard_pslice(scores, percents, higher_is_better)
        printf(f'Percentile')
        printf(f'{"Score":>{8}}')
        print()
        for i in range(len(x)):
            printf(f'{x[i]:>{10}}')
            printf(f'{y[i]:>{8}.{n_decimals}f}')
            print()
    if show_plot:
        more_percents = seq(min(percents), max(percents), step = 2.5)
        x, y = leaderboard_pslice(scores, more_percents, higher_is_better)
        plot_leaderboard(x, y, score, higher_is_better)
        pyplot.axhline(numpy.percentile(scores, plot_percents), color = 'b')


def leaderboard_rank(
        score: float,
        scores: 'Iterable[float]',
        higher_is_better: bool = True) \
        -> int:
    rank = abs(
            higher_is_better * len(scores) - numpy.searchsorted(scores, score))
    return rank


def leaderboard_pslice(
        scores: 'Iterable[float]',
        percents: 'Iterable[float]' = None,
        higher_is_better = True) \
        -> 'Tuple[Iterable[float], Iterable[float]]':
    if percents is None:
        percents = seq(100, 10, -5)
    ptiles = pslice_top(scores, percents, higher_is_better)
    return percents, ptiles


def leaderboard_ptile(
        score: float,
        scores: 'Iterable[float]',
        higher_is_better: bool = True) \
        -> float:
    raw = percentileofscore(scores, score, kind = 'mean')
    return abs(100 * (not higher_is_better) - raw)


def read_leaderboard(
        path = 'leaderboard.csv',
        col_name = 'Score') \
        -> 'ndarray':
    df_col = pandas.read_csv(path, usecols = [col_name])
    return numpy.sort(df_col.values.ravel())


def plot_leaderboard(
        percents: 'Iterable[float]',
        percentiles: 'Iterable[float]',
        score: float = None,
        higher_is_better: bool = True,
        figsize: 'Tuple[int, int]' = None,
        fontsize: int = 16) \
        -> None:
    if figsize is None:
        figsize = (PYPLOT_WIDTH, PYPLOT_HEIGHT * 0.4)
    pyplot.figure(figsize = figsize)
    pyplot.xlabel('Percent', fontsize = fontsize)
    pyplot.ylabel('Score', fontsize = fontsize)
    if score is not None:
        pyplot.axhline(score, color = 'r')
    seaborn.pointplot(percents, percentiles)
    if not higher_is_better:
        pyplot.gca().invert_yaxis()

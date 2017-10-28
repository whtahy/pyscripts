# Kaggle utils
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/

from typing import TYPE_CHECKING

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from pyscripts.globals import PYPLOT_HEIGHT, PYPLOT_WIDTH
from pyscripts.knot import printf
from pyscripts.numbers import pslice_top, seq
from scipy.stats import percentileofscore

if TYPE_CHECKING:
    from typing import *
    from pyscripts.types import *


def leaderboard(
        percents: 'Iterable[float]' = None,
        score: int = None,
        n_decimals: int = 0,
        higher_is_better: bool = True,
        show_rank: bool = True,
        show_ptile: bool = True,
        show_pslice: bool = True,
        show_plot: bool = True,
        width: int = 24) -> None:
    scores = read_leaderboard()
    if percents is not None:
        percents = numpy.sort(percents)[::-1]
    if show_ptile:
        ptile = leaderboard_ptile(score, scores, higher_is_better)
        print(f'Percentile: {ptile:.1f}%')
    if show_rank:
        rank = leaderboard_rank(score, scores, higher_is_better)
        print(f'Rank: {rank} of {len(scores)}')
    if show_pslice:
        x, y = leaderboard_pslice(scores, percents, higher_is_better)
        print()
        printf(f'Percentile')
        printf(f'{"Score":>8}')
        print()
        for i in range(len(x)):
            printf(f'{x[i]:>10}')
            printf(f'{y[i]:>8.{n_decimals}f}')
            print()
    if show_plot:
        x, y = leaderboard_pslice(scores, percents, higher_is_better)
        plot_leaderboard(x, y, score, higher_is_better)


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

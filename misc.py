# Misc utils
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import Any, Iterable, List, Union

import numpy
from pandas import read_csv
from pyscripts.numbers import seq
from scipy.stats import percentileofscore as percentileof


def leaderboard(
        score = None,
        file_name = 'leaderboard.csv',
        col = 'Score',
        metric = 'Percentile',
        higher_is_better = True,
        step = 20,
        padding = 4,
        decimals = 0) \
        -> None:
    if higher_is_better:
        percents = seq(100, step, -step)
    else:
        percents = seq(0, 100 - step, step)
    scores = read_csv(file_name)[col]
    width_left = len(metric)
    width_right = len(str(f'{numpy.median(scores):.{decimals}f}')) + padding
    if score is not None:
        quantile = abs(
                100 * (not higher_is_better) - percentileof(scores, score))
        print(f"{'Rank':>{width_left}}"
              f"{quantile:>{width_right-1}.{decimals}f}%\n")
    print(f"{metric:>{width_left}}{'Score':>{width_right}}")
    for i in range(0, len(percents)):
        left = 100 - i * step
        right = numpy.percentile(scores, percents[i])
        if len(f"{right:.{decimals}f}") > width_right:
            print(f"{left:>{width_left}}{'--':>{width_right}}")
        else:
            print(f"{left:>{width_left}}{right:>{width_right}.{decimals}f}")


def loc_of(
        *objects: Any) \
        -> Union(int, List[int]):
    locs = [id(obj) for obj in objects]
    if len(locs) == 1:
        return locs[0]
    else:
        return locs


def subdict(full_dict: dict, keys: Iterable[Any]):
    return {k: v for k, v in full_dict.items() if k in keys}

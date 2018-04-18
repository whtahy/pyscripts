# Plot utils
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from math import ceil
from typing import Sequence

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

MAX_WIDTH = 24
MAX_HEIGHT = 20


def plot_confmat(
        y: pd.Series,
        y_hat: Sequence,
        rotate_x: int = 0,
        rotate_y: int = 'vertical') \
        -> None:
    """
    Plot confusion matrix using `seaborn`.
    """
    classes = y.unique()
    ax = sns.heatmap(
        confusion_matrix(y, y_hat),
        xticklabels=classes,
        yticklabels=classes,
        annot=True,
        square=True,
        cmap="Blues",
        fmt='d',
        cbar=False)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=rotate_x)
    plt.yticks(rotation=rotate_y, va='center')
    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')


#
# Helpers

def plotgrid_dims(
        n_items) \
        -> 'Tuple[int, int]':
    """
    No. of plots per column (`nrows`) and plots per row (`ncols`)
    """
    ncols = plotgrid_ncols(n_items)
    nrows = int(ceil(n_items / ncols))
    return nrows, ncols


def plotgrid_ncols(
        n_items: int) \
        -> int:
    """
    No. of plots per row.
    """
    default = max(1, int(ceil(n_items ** 0.5)))
    return {
        1: 1,
        2: 2,
        3: 3,
        4: 2,
        5: 3,
        6: 3
    }.get(n_items, default)


def plotgrid_nrows(
        n_items: int) \
        -> int:
    """
    No. of plots per column.
    """
    return plotgrid_dims(n_items)[0]


def fig_dims(
        n_items: int,
        max_width: int = MAX_WIDTH,
        max_height: int = MAX_HEIGHT) \
        -> 'Tuple[int, int]':
    """
    `figsize` for plot grid.
    """
    return fig_width(n_items, max_width), fig_height(n_items, max_height)


def fig_height(
        n_items: int,
        max_height: int = MAX_HEIGHT) \
        -> int:
    """
    Figure height adjusted for no. of plots per column.
    """
    nrows = plotgrid_nrows(n_items)
    if nrows == 1:
        return round(max_height * 0.4)
    elif nrows == 2:
        return round(max_height * 0.6)
    else:
        return max_height


def fig_width(
        n_items: int,
        max_width: int = MAX_WIDTH) \
        -> int:
    """
    Figure width adjusted for no. of plots per row.
    """
    ncols = plotgrid_ncols(n_items)
    if ncols == 1:
        return round(max_width * 0.4)
    elif ncols == 2:
        return round(max_width * 0.6)
    else:
        return max_width

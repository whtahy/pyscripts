# Plot utils
# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


from typing import TYPE_CHECKING

import matplotlib.pyplot as pyplot
import numpy
import seaborn
from numpy import ceil, mean, median

from pyscripts.commune import PYPLOT_HEIGHT, PYPLOT_WIDTH
from pyscripts.zfc import numpy_ncols, pslice

if TYPE_CHECKING:
    from typing import *
    from pyscripts.mytypes import *


def plot_err(
        y: 'ndarray',
        y_hat: 'ndarray',
        y_name: str = 'y',
        figsize: 'NLT_IntType' = None,
        mask = True,
        sample_size: int = None) \
        -> None:
    if figsize is None:
        figsize = (PYPLOT_WIDTH // 2, PYPLOT_HEIGHT * 0.7)

    if sample_size is None:
        sample_size = min(1000, len(y))
    idx = pslice(numpy.arange(len(y)), n_elements = sample_size).astype('int')
    y = y[idx].ravel()
    y_hat = y_hat[idx].ravel()

    err = y_hat - y
    abs_err = abs(err)

    fig, axs = pyplot.subplots(3, 2, figsize = figsize)

    axs[0, 0].set_xlabel(f'Predicted {y_name}')
    axs[0, 0].set_ylabel('Error')
    axs[0, 0].axhline(y = mean(err), color = 'r')
    axs[0, 0].axhline(color = 'b')
    axs[0, 0].scatter(y_hat[mask], err[mask])

    axs[0, 1].set_xlabel(y_name)
    axs[0, 1].set_ylabel('Error')
    axs[0, 1].axhline(y = mean(err), color = 'r')
    axs[0, 1].axhline(color = 'b')
    axs[0, 1].scatter(y[mask], err[mask])

    axs[1, 0].set_xlabel(f'Predicted {y_name}')
    axs[1, 0].set_ylabel('Abs Error')
    axs[1, 0].axhline(y = mean(abs_err), color = 'r')
    axs[1, 0].axhline(color = 'b')
    axs[1, 0].scatter(y_hat[mask], abs_err[mask])

    axs[1, 1].set_xlabel(y_name)
    axs[1, 1].set_ylabel('Abs Error')
    axs[1, 1].axhline(y = mean(abs_err), color = 'r')
    axs[1, 1].axhline(color = 'b')
    axs[1, 1].scatter(y[mask], abs_err[mask])

    seaborn.distplot(err, ax = axs[2, 0])
    axs[2, 0].set_xlabel(f'Error')
    seaborn.distplot(abs_err, ax = axs[2, 1])
    axs[2, 1].set_xlabel(f'Abs Error')


def plot_arrays(
        arrays: 'ndarray',
        plot_func: 'Callable[[int], int]' = seaborn.distplot,
        titles: bool = None,
        draw_func: bool = True,
        func: 'Callable[[numpy.ndarray], float]' = median) \
        -> None:
    n_plots = len(arrays)
    nrows, ncols = plotgrid_dims(n_plots)
    figsize = fig_dims(n_plots)
    fig, axs = pyplot.subplots(nrows = nrows,
                               ncols = ncols,
                               figsize = figsize)
    for coord in numpy.ndenumerate(axs):
        if nrows > 1:
            row = coord[0][0]
            col = coord[0][1]
            ax = axs[row, col]
        else:
            row = 0
            col = coord[0][0]
            ax = axs[col]
        index = row * ncols + col
        if titles is not None:
            ax.set_title(titles[index])
        data = arrays[index]
        plot_func(data, ax = ax)
        if draw_func:
            loc = func(data)
            ax.axvline(x = loc)
        if index + 1 >= n_plots:
            break


def plotgrid_arr(
        arr: 'ndarray',
        plot_func = seaborn.distplot,
        names: 'NLT_StrType' = None,
        figsize: 'Tuple[int, int]' = None,
        func: 'Callable[NLT_FloatType, float]' = median,
        draw_func: bool = True,
        sample_size: int = None) \
        -> None:
    if sample_size is None:
        sample_size = min(1000, len(arr))
    idx = pslice(numpy.arange(len(arr)), n_elements = sample_size).astype('int')
    arr = arr[idx]

    n_plots = numpy_ncols(arr)
    nrows, ncols = plotgrid_dims(n_plots)
    if figsize is None:
        figsize = fig_dims(n_plots)
    fig, axs = pyplot.subplots(nrows = nrows,
                               ncols = ncols,
                               figsize = figsize)
    for index in range(0, n_plots):
        row = index // ncols
        col = index % ncols
        if nrows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]
        if names is not None:
            ax.set_title(names[index])
        data = arr[:, index]
        plot_func(data, ax = ax)
        if draw_func:
            loc = func(data)
            ax.axvline(x = loc)


#
#
# Helpers


def plotgrid_dims(
        n_items) \
        -> 'Tuple[int, int]':
    ncols = plotgrid_ncols(n_items)
    nrows = int(ceil(n_items / ncols))
    return nrows, ncols


def fig_dims(
        n_items: int,
        max_width: int = PYPLOT_WIDTH,
        max_height: int = PYPLOT_HEIGHT) \
        -> 'Tuple[int, int]':
    return fig_width(n_items, max_width), fig_height(n_items, max_height)


def plotgrid_ncols(
        n_items: int) \
        -> int:
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
    return plotgrid_dims(n_items)[0]


def fig_height(
        n_items: int,
        max_height: int = PYPLOT_HEIGHT) \
        -> int:
    nrows = plotgrid_nrows(n_items)
    if nrows == 1:
        return round(max_height * 0.4)
    elif nrows == 2:
        return round(max_height * 0.6)
    else:
        return max_height


def fig_width(
        n_items: int,
        max_width: int = PYPLOT_WIDTH) \
        -> int:
    ncols = plotgrid_ncols(n_items)
    if ncols == 1:
        return round(max_width * 0.4)
    elif ncols == 2:
        return round(max_width * 0.6)
    else:
        return max_width

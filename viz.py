# Plot utils
# Released under CC0:
# https://creativecommons.org/publicdomain/zero/1.0/


from typing import TYPE_CHECKING

import matplotlib.pyplot as pyplot
import numpy
import seaborn
from numpy import ceil, mean, median
from pyscripts.globals import PYPLOT_HEIGHT, PYPLOT_WIDTH
from pyscripts.numbers import percentiles

if TYPE_CHECKING:
    from typing import Callable, Iterable


def plot_err(y: numpy.ndarray,
             y_hat: numpy.ndarray,
             y_name: str = 'y',
             figsize: Iterable[int] = (PYPLOT_WIDTH // 2 + 1,
                                       PYPLOT_WIDTH // 2 + 1),
             mask = True):
    err = y_hat - y
    abs_err = abs(err)

    fig, axs = pyplot.subplots(3, 2, figsize = figsize)

    axs[0, 0].set_xlabel(y_name)
    axs[0, 0].set_ylabel('Error')
    axs[0, 0].axhline(color = 'r')
    axs[0, 0].scatter(y[mask], err[mask])

    axs[0, 1].set_xlabel(f'Predicted {y_name}')
    axs[0, 1].set_ylabel('Error')
    axs[0, 1].axhline(color = 'r')
    axs[0, 1].scatter(y_hat[mask], err[mask])

    axs[1, 0].set_xlabel(y_name)
    axs[1, 0].set_ylabel('Abs Error')
    axs[1, 0].axhline(y = mean(abs_err), color = 'b')
    axs[1, 0].scatter(y[mask], abs_err[mask])

    axs[1, 1].set_xlabel(f'Predicted {y_name}')
    axs[1, 1].set_ylabel('Abs Error')
    axs[1, 1].axhline(y = mean(abs_err), color = 'b')
    axs[1, 1].scatter(y_hat[mask], abs_err[mask])

    seaborn.distplot(err, ax = axs[2, 0])
    axs[2, 0].set_xlabel(f'Error')
    seaborn.distplot(abs_err, ax = axs[2, 1])
    axs[2, 1].set_xlabel(f'Abs Error')


# REFACTOR
def plot_arrays(arrays: numpy.ndarray,
                plot_func: Callable[[int], int] = seaborn.distplot,
                titles: bool = None,
                sample: bool = True,
                sample_threshold: int = 5000,
                fig_width: int = PYPLOT_WIDTH,
                fig_height: int = PYPLOT_HEIGHT,
                draw_func: bool = True,
                func: Callable[[numpy.ndarray], float] = median):
    n_plots = len(arrays)
    ncols = int(max(ceil(numpy.sqrt(n_plots)), min_cols))
    nrows = ceil(n_plots / ncols).astype('int')
    if nrows <= 1:
        figsize = (figsize[0] // (1 + (ncols == 2)), figsize[1] // 3)
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
        if sample:
            data = percentiles(arrays[index])
        else:
            data = arrays[index]
        plot_func(data, ax = ax)
        if draw_func:
            loc = func(data)
            ax.axvline(x = loc)
        if index + 1 >= n_plots:
            break


# REFACTOR
def plot_array(numpy_array,
               plot_func = seaborn.distplot,
               titles = None,
               xlabels = None,
               ylabels = None,
               sample = True,
               sample_threshold = 5000,
               sharex = False,
               sharey = False,
               draw_func = True,
               func = median,
               fig_height = PYPLOT_HEIGHT,
               fig_width = PYPLOT_WIDTH):
    n_plots = numpy_array.shape[1]
    ncols = max(ceil(numpy.sqrt(n_plots)), min_cols).astype('int')
    nrows = ceil(n_plots / ncols).astype('int')
    if nrows <= 1:
        figsize = (figsize[0] // (1 + (ncols == 2)), figsize[1] // 3)
    fig, axs = pyplot.subplots(nrows = nrows,
                               ncols = ncols,
                               sharex = sharex,
                               sharey = sharey,
                               figsize = figsize)
    for index in range(0, n_plots):
        row = index // ncols
        col = index % ncols
        if nrows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]
        if titles is not None:
            ax.set_title(titles[index])
        if sample:
            data = percentiles(numpy_array[index])
        else:
            data = numpy_array[index]
        data = numpy_array[:, index]
        plot_func(data, ax = ax)
        if draw_func:
            loc = median(data)
            ax.axvline(x = loc)

from __future__ import print_function, division
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import sys, os, re, glob, argparse, parse, ast, shutil
from collections import defaultdict, OrderedDict
from functools import lru_cache
import numpy as np
import scipy.stats as stats
np.random.seed(0)
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .common import (
    get_terminal_size,
    non_string_iterable,
    as_non_string_iterable
)


def plot(
    df,
    x=None,
    y=None,
    hue=None,
    grouped=True,
    block=None,
    block_levels=None,
    block_orient='h',
    height=3,
    width=3,
    n_cols=None,
    xlim={},
    ylim={},
    plot_func=sns.pointplot,
    plot_kws={},
    legend=True,
    legend_row=-1,
    legend_col=None,
    legend_kws={},
    verbose=False,
    tight=True,
    gridspec_kws={},
):
    df = df.copy()

    # establish variables
    if x is None:
        x = [p for p in df.index.names if p != 'job_name']
    x = as_non_string_iterable(x)

    if y is None:
        y = df.columns
    y = as_non_string_iterable(y)

    if grouped: # for each x var, group by every other hue var
        if hue is None:
            hue = x
        hue = as_non_string_iterable(hue)
        grouped_hues = dict()
        for i, x_i in enumerate(x):
            grouped_hues[x_i] = add_group_column(
                df, [h_j for h_j in hue if h_j != x_i]
            )

    elif non_string_iterable(hue):
        hue = add_group_column(df, list(hue))

    if verbose:
        print(f'x = {x}')
        print(f'y = {y}')
        print(f'hue = {hue}')
        print(f'grouped = {grouped}')

    if block is not None:
        if block_levels is None:
            block_levels = df[block].unique()
        n_blocks = len(block_levels)
        assert block_orient in {'h', 'v'}
        if block_orient == 'h':
            x *= n_blocks
        else:
            y *= n_blocks

    legend_defaults = dict(
        loc='upper left',
        bbox_to_anchor=(0, -0.2),
        frameon=False,
    )
    legend_defaults.update(legend_kws)
    legend_kws = legend_defaults

    df = df.reset_index()
    assert len(df) > 0, 'empty data frame'

    n_rows, n_cols = get_n_rows_and_cols(x, y, n_cols)

    if legend_row is not None:
        legend_row = as_array_idx(legend_row, n_rows)
    if legend_col is not None:
        legend_col = as_array_idx(legend_col, n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(width*n_cols, height*n_rows),
        squeeze=False,
        gridspec_kw=gridspec_kws,
    )
    iter_axes = iter(axes.flatten())

    # track whether any rows/columns all have the same x/y data
    row_ys = defaultdict(set)
    col_xs = defaultdict(set)

    for i, y_i in enumerate(y):

        for j, x_j in enumerate(x):

            ax = next(iter_axes)
            col_idx = (i*len(x) + j) % n_cols
            row_idx = (i*len(x) + j) // n_cols
            row_ys[row_idx].add(y_i)
            col_xs[col_idx].add(x_j)

            if j == 0:
                sharey_ax = ax

            if grouped:
                hue = grouped_hues[x_j]

            curr_df = df
            if block is not None:
                if block_orient == 'h':
                    block_idx = col_idx // (n_cols // n_blocks)
                    if row_idx == 0:
                        ax.set_title(block_levels[block_idx])
                else:
                    block_idx = row_idx // (n_rows // n_blocks)
                    if col_idx == 0:
                        ax.set_title(block_levels[block_idx])
                curr_df = df[df[block] == block_levels[block_idx]]

            if verbose:
                print((y_i, x_j), (row_idx, col_idx))

            plot_func(data=curr_df, x=x_j, y=y_i, hue=hue, ax=ax, **plot_kws)

            if ax.legend_:
                ax.legend_.remove()

            if (
                legend and
                (legend_row is None or row_idx == legend_row) and
                (legend_col is None or col_idx == legend_col)
            ):
                handles, labels = ax.get_legend_handles_labels()
                label_map = OrderedDict(zip(labels, handles))

                legend_kws_copy = legend_kws.copy()
                if 'title' not in legend_kws:
                    legend_kws_copy['title'] = hue

                ax.legend(
                    label_map.values(),
                    label_map.keys(),
                    **legend_kws_copy,
                )

            if x_j in xlim:
                ax.set_xlim(*xlim[x_j])

            if y_i in ylim:
                ax.set_ylim(*ylim[y_i])

    # only show left-most y label on rows with the same y data
    for row_idx in range(n_rows):
        if len(row_ys[row_idx]) < 2:
            for col_idx in range(1, n_cols):
                ax = axes[row_idx,col_idx]
                ax.set_ylabel(None)
                ax.set_yticklabels([])

    # only show bottom-most x label on columns with the same x data
    for col_idx in range(n_cols):
        if len(col_xs[col_idx]) < 2:
            for row_idx in range(n_rows-1):
                ax = axes[row_idx,col_idx]
                ax.set_xlabel(None)
                ax.set_xticklabels([])

    # turn off remaining axes that have nothing plotted
    for ax in iter_axes:
        ax.axis('off')

    for ax in axes.flatten():
        ax.set_axisbelow(True) # grid lines behind data
        ax.grid(True, linestyle=':', color='lightgray')

    sns.despine(fig)

    sns.despine(top=True, right=True)
    if tight:
        fig.tight_layout()
    return fig


def get_n_rows_and_cols(x, y, n_cols=None):
    if n_cols is None:
        n_cols = len(x)
    n_axes = len(x) * len(y)
    assert n_axes > 0
    n_rows = (n_axes + n_cols - 1)//n_cols
    n_cols = min(n_axes, n_cols)
    return n_rows, n_cols


@lru_cache(100)
def make_group_value_from_tuple(tup):
    #tup = tuple('{:.1e}'.format(v) if isinstance(v, float) else v for v in tup)
    return str(tup).replace('False', '0').replace('True', '1')


def make_group_value(values):
    return make_group_value_from_tuple(tuple(values))


def add_group_column(df, group_cols, do_print=False):
    '''
    Add a new column to df that combines the values
    in group_cols columns into tuple strings.
    '''
    if len(group_cols) == 1:
        return group_cols[0]
    group = '({})'.format(', '.join(group_cols))
    if do_print:
        print('adding group column {}'.format(repr(group)))
    df[group] = df[group_cols].apply(make_group_value, axis=1)
    return group


def as_array_idx(i, n):
    '''
    Convert i to an index into an array
    of length n, i.e. negative values
    index from the end of the array.
    '''
    return n + i if i < 0 else i


def get_palette(
    n_hues,
    n_shades=1,
    n_repeat=1,
    hues=None,
    min_val=0.0,
    max_val=1.0,
    n_samples=100,
    mode=None,
    reverse=False,
):
    assert 0 <= min_val <= 1.0
    assert 0 <= max_val <= 1.0
    
    if hues is None:
        if n_hues <= 9:
            mode = mode or 'muted'
            hues = sns.color_palette(mode)[:n_hues]
        else:
            mode = mode or 'husl'
            hues = sns.color_palette(mode, n_hues)
            
    if not isinstance(n_shades, list):
        n_shades = [n_shades] * len(hues)

    colors = []
    for hue, n_shades in zip(hues, n_shades):
        
        # get n_samples different shades of hue
        shades = (
            sns.dark_palette(hue, n_colors=n_samples//2) + \
            sns.light_palette(hue, n_colors=n_samples//2, reverse=True)
        )

        # limit shade range with min_val and max_val
        min_idx = int(min_val * len(shades))
        max_idx = int(max_val * len(shades))
        shades = shades[min_idx:max_idx]
  
        # get n_shades evenly spaced shades in that range, avoiding endpoints
        vals = np.linspace(0, 1, n_shades + 2)
        idxs = [int(v * (len(shades) - 1)) for v in vals]
        shades = [shades[i] for i in idxs[1:-1]]
        
        # repeat each shade n_repeat times
        shades = sorted(n_repeat * shades, key=lambda x: sum(x))

        if reverse: # reverse the shading order
            shades = shades[::-1]

        colors.extend(shades)
        
    return sns.color_palette(colors)



def annotate_pearson_r(x, y, **kwargs):
    print(kwargs)
    nan = np.isnan(x) | np.isnan(y)
    r, _ = stats.pearsonr(x[~nan], y[~nan])
    plt.gca().annotate("$\\rho = {:.2f}$".format(r), xy=(.5, .8),
        xycoords='axes fraction', ha='center', fontsize='large')


def my_dist_plot(a, **kwargs):
    if 'label' in kwargs:
        kwargs['label'] = str(kwargs['label'])
    return sns.distplot(a[~np.isnan(a)], **kwargs)


def plot_corr(plot_file, df, x, y, hue=None, height=4, width=4, dist_kws={}, scatter_kws={}, **kwargs):

    df = df.reset_index()
    g = sns.PairGrid(df, x_vars=x, y_vars=y, hue=hue, height=height, aspect=width/float(height), **kwargs)
    g.map_diag(my_dist_plot, **dist_kws)
    g.map_offdiag(plt.scatter, **scatter_kws)
    #g.map_upper(sns.kdeplot, shade=True)
    #g.map_offdiag(annotate_pearson_r)
    fig = g.fig
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(plot_file, bbox_inches='tight')
    #plt.close(fig)
    return fig

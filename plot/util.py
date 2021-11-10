import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import matplotlib.colors as mcolors
from util import is_outlier

_tableaeu_colors = list(mcolors.TABLEAU_COLORS.keys())

def plot_histogram(values, bins=20, logscale=False, eps=1e-10):
    """ Plots something as a histogram. 
    
    Parameters:
    -----------
    values : ndarray, shape [N]
        Values to create a histogram of.
    bins : int
        Bins argument for `plt.hist` function.
    logscale : bool
        If True, the values are binned in logspace.
    eps : float
        Small value to add to the values to bin in logspace to prevent infinities.
    
    Returns:
    --------
    plt.Figure
        The plot.
    plt.axis.Axes
        The axis of the plot.
    """
    fig, ax = plt.subplots(1, 1)
    if logscale:
        values += eps # Small eps
        bins = np.histogram_bin_edges(values, bins=bins)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        ax.hist(values, bins=logbins)
        ax.set_xscale('log')
    else:
        ax.hist(values, bins=bins)
    return fig, ax


def plot_histograms(x, labels, label_names=None, bins=20, log_scale=False, kind='vertical', kde=True, x_label='density', y_label='label'):
    """ Plots an histogram for data of different labels.
    
    Parameters:
    -----------
    x : torch.tensor, shape [N]
        Values to plot a histogram of.
    labels : torch.tensor, shape [N]
        Label for each point.
    label_names : dict or None
        Name for each label in the legend. If None is given, it is set to the label's idx.
    log_scale : bool or int
        The logarithmic scale on the x axis.
    kind : str
        Supported modes are:
        - 'vertical' : Each label get its own color-map row that indicates the density
        - 'overlapping' : Normal overlapping histograms.
    kde : bool
        If True, a kernel-density-estimation is applied as well.

    
    Returns:
    --------
    plt.Figure
        The plot.
    plt.axis.Axes
        The axis of the plot or an array of axis if `overlapping` is set to `False`.
    """
    if label_names is None:
        label_names = {label : f'Label {label}' for label in np.unique(labels)}

    df = pd.DataFrame({
        x_label : x.cpu().numpy(),
        y_label : [label_names[label] for label in labels.cpu().numpy()]
    })
    fig, ax = plt.subplots(1, 1)
    if kind.lower() == 'vertical':
        sns.histplot(df, x=x_label, y=y_label, hue=y_label, 
                     ax=ax, bins=bins, kde=kde, stat='density', 
                     common_norm=False, log_scale=(log_scale, False), 
                     hue_order=[label_names[label] for label in sorted(np.unique(labels))], discrete=(False, True))
    elif kind.lower() == 'overlapping':
        sns.histplot(df, x=x_label, hue=y_label,
                     element='step',
                     ax=ax, bins=bins, kde=kde, stat='density', 
                     common_norm=False, log_scale=(log_scale, False), 
                     hue_order=[label_names[label] for label in sorted(np.unique(labels))])
    else:
        raise RuntimeError(f'Density plot kind {kind} not supported.')
    return fig, ax

def plot_2d_histogram(x, y, bins=20, x_label='x', y_label='y', log_scale_x=False, log_scale_y=False, kde=False):
    """ Plots a 2d histogram (heatmap)
    
    Parameters:
    -----------
    x : torch.tensor, shape [N]
        X values to plot.
    y : torch.tensor, shape [N]
        Y values to plot.
    x_label : str
        Name for x values.
    y_label : str
        Name for y values.
    log_scale_x : bool or int
        The logarithmic scale on the x axis.
    log_scale_y : bool or int
        The logarithmic scale on the y axis.
    kde : bool
        If True, a kernel-density-estimation is applied as well.

    
    Returns:
    --------
    plt.Figure
        The plot.
    plt.axis.Axes
        The axis of the plot or an array of axis if `overlapping` is set to `False`.
    """
    df = pd.DataFrame({
        x_label : x.cpu().numpy(),
        y_label : y.cpu().numpy()
    })
    fig, ax = plt.subplots(1, 1)
    sns.histplot(df, x=x_label, y=y_label, ax=ax, bins=bins, kde=kde, 
                 stat='density', common_norm=False, log_scale=(log_scale_x, log_scale_y))
    return fig, ax
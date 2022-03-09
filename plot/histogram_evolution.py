# Plots the evolution of a histogram as 2d heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Tuple, Any

def plot_histogram_evolution(values: np.ndarray, num_bins: int = 25, figsize: Tuple[float, float]=(10, 5), value_name: Optional[str]=None,
        time_name: Optional[str]='Epoch', time_interval: int=1) -> Tuple[plt.Figure, Any]:
    """ Plots the evolution of a 1d distribution as a 2d heatmap with time on the x axis.
    
    Parameters:
    -----------
    values : ndarray, shape [sequence_length, num_observations]
        The observations to plot the distribution of.
    num_bins : int, optional, default: 25
        How many bins to use for the observations.
    figsize : Tuple[float, float], optional, default: (10, 5)
        The size of the plt figure.
    value_name : str, optional, default: None
        If given, what the observations are called.
    time_name : str, optional, default: 'Epoch'
        If given, what the temporal axis should be labeled with.
    time_interval : int, optional, default: 1
        The step in which the temporal x axis goes.
    
    Returns:
    --------
    fig : plt.Figure
        The figure.
    ax : plt.axes.Axes
        The axis.
    """ 
    bin_centers = np.linspace(np.round(values.min() - 0.5, 0), np.round(values.max() + 0.5, 0), num_bins)

    distances = np.abs(values.flatten()[:, None] - bin_centers[None, :])
    bin_idxs = distances.argmin(1).reshape(values.shape)

    counts = np.zeros((num_bins, values.shape[0]))
    for epoch in range(values.shape[0]):
        for bin_idx in bin_idxs[epoch]:
            counts[bin_idx, epoch] += 1

    counts = pd.DataFrame(counts, index=np.round(bin_centers, 2), columns=[i * time_interval for i in range(values.shape[0])])
    if value_name:
        counts.index.name = value_name
    if time_name:
        counts.columns.name = time_name

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(counts, robust=True, ax=ax)
    return fig, ax

def plot_heatmap2d(values: np.ndarray, figsize: Tuple[float, float]=(10, 5), value_name: Optional[str]=None,
        time_name: Optional[str]='Epoch', time_interval: int=1, log_scale: bool=True, cbar_label: Optional[str]=None) -> Tuple[plt.Figure, Any]:
    """ Plots a 2d heatmap.
    
    Parameters:
    -----------
    values : ndarray, shape [sequence_length, num_observations]
        The observations to plot the distribution of.
    figsize : Tuple[float, float], optional, default: (10, 5)
        The size of the plt figure.
    value_name : str, optional, default: None
        If given, what the observations are called.
    time_name : str, optional, default: 'Epoch'
        If given, what the temporal axis should be labeled with.
    time_interval : int, optional, default: 1
        The step in which the temporal x axis goes.
    
    Returns:
    --------
    fig : plt.Figure
        The figure.
    ax : plt.axes.Axes
        The axis.
    """ 
    if log_scale:
        values = np.log(values)
        if cbar_label:
            cbar_label = f'Log {cbar_label}'
    fig, ax = plt.subplots(figsize=figsize)
    df = pd.DataFrame(values.T, columns=time_interval * np.arange(values.shape[0]))
    if value_name:
        df.index.name = value_name
    if time_name:
        df.columns.name = time_name
    if cbar_label:
        cbar_kwargs = {
            'label' : cbar_label
        }
    else:
        cbar_kwargs = {}
    sns.heatmap(df, ax=ax, cbar_kws=cbar_kwargs)
    return fig, ax
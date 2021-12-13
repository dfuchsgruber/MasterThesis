import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import matplotlib.colors as mcolors
from util import is_outlier
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_2d

_tableaeu_colors = list(mcolors.TABLEAU_COLORS.keys())


def plot_density(points_to_fit, points_to_eval, density_model, labels, label_names, seed=1337, bins=100, levels=20):
    """ Creates a density plot for high dimensional points by using 2d PCA. Also visualizes the density by creating a meshgrid in 2d pca space
    and using the inverse transform to find densities for these points.
    
    Parameters:
    -----------
    points_to_fit : torch.Tensor, shape [N, D]
        The points the density model was fit to.
    points_to_eval : torch.Tensor, shape [N', D]
        The points the density model is evaluated on.
    density_model : nn.Module
        A callable module that evaluates the density for points of shape (*, D)
    labels : torch.Tensor, shape [N']
        Labels for the points to evaluate.
    label_names : dict
        A  mapping that names the points to evaluate.
    seed : int
        The seed for the PCA.
    bins : int
        How many bins to use for the meshgrid.
    levels : int
        How many contour levels to plot.

    Returns:
    --------
    plt.Figure
        The plot.
    plt.axis.Axes
        The axis of the plot.
    """
    
    bins_x, bins_y = bins, bins
    bins_x = 7
    
    # First create a PCA and embed points to fit
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(points_to_fit.cpu().numpy())
        
    # Embed the points to eval
    points_to_eval_emb = pca.transform(points_to_eval.cpu().numpy())
    mins, maxs = points_to_eval_emb.min(0), points_to_eval_emb.max(0)
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], bins_x), np.linspace(mins[1], maxs[1], bins_y), indexing='ij')
    
    
    # Flatten
    xx, yy = xx.reshape((-1, 1)), yy.reshape((-1, 1))
    grid = np.concatenate((xx, yy), axis=-1)
    
    # Get the densities
    density_points = density_model(points_to_eval).cpu().numpy()
    density_grid = density_model(torch.tensor(pca.inverse_transform(grid)).float()).cpu().numpy()
    
    xx, yy, density_grid = xx.reshape((bins_x, bins_y)), yy.reshape((bins_x, bins_y)), density_grid.reshape((bins_x, bins_y))
    
    fig, ax = plt.subplots(1, 1)
    c = ax.contourf(xx, yy, density_grid, cmap='Reds', levels=levels)
    fig.colorbar(c, ax=ax)
    #ax.scatter(samples_emb[:, 0], samples_emb[:, 1], c=density_samples)
    for label, name in label_names.items():
        points_to_plot = points_to_eval_emb[labels == label]
        ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], label=name, marker='x')
    ax.legend()
    return fig, ax

def plot_density_sampling(points_to_fit, points_to_eval, density_model, labels, label_names, seed=1337, num_samples=10000, bins=25):
    """ Creates a density plot for high dimensional points by using 2d PCA. Also visualizes the density by sampling `num_samples` points
        in a box that contains all `points_to_eval` in the high dimensional space, evaluating their density and projecting down into the 
        2d space. The downprojected densities are binned and averaged.
    
    Parameters:
    -----------
    points_to_fit : torch.Tensor, shape [N, D]
        The points the density model was fit to.
    points_to_eval : torch.Tensor, shape [N', D]
        The points the density model is evaluated on.
    density_model : nn.Module
        A callable module that evaluates the density for points of shape (*, D)
    labels : torch.Tensor, shape [N']
        Labels for the points to evaluate.
    label_names : dict
        A  mapping that names the points to evaluate.
    seed : int
        The seed for the PCA.
    num_samples : int
        How many random samples should be drawn to estimate the density in lower dimensional space.
    bins : int
        How many bins to use when digitzing downprojected density samples.

    Returns:
    --------
    plt.Figure
        The plot.
    plt.axis.Axes
        The axis of the plot.
    """
    
    # First create a PCA and embed points to fit
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(points_to_fit.cpu().numpy())
        
    # Randomly sample some points within a compact volume that contains evaluation points and sample from regions that contains fitting points
    samples = torch.rand((2 * num_samples, points_to_eval.size(1)))
    lower, upper = points_to_eval.min(0)[0], points_to_eval.max(0)[0]
    samples[:num_samples] *= (upper - lower)
    samples[:num_samples] += lower
    lower, upper = points_to_fit.min(0)[0], points_to_fit.max(0)[0]
    samples[num_samples:] *= (upper - lower)
    samples[num_samples:] += lower
    
    # Get the densities
    density_samples = density_model(samples).numpy()
    # density_points = density_model(points_to_eval).cpu().numpy()
    
    # Embed the points to eval
    points_to_eval_emb = pca.transform(points_to_eval.cpu().numpy())
    samples_emb = pca.transform(samples.cpu().numpy())
    
    binned_density, bins_x, bins_y, _ = binned_statistic_2d(
        samples_emb[:, 0], samples_emb[:, 1], density_samples, bins=bins)
    bins_x, bins_y = 0.5 * (bins_x[1:] + bins_x[:-1]), 0.5 * (bins_y[1:] + bins_y[:-1])
    bins_xx, bins_yy = np.meshgrid(bins_x, bins_y) 
    
    fig, ax = plt.subplots(1, 1)
    c = ax.contourf(bins_xx, bins_yy, binned_density.T, cmap='Reds', levels=levels)
    fig.colorbar(c, ax=ax)
    #ax.scatter(samples_emb[:, 0], samples_emb[:, 1], c=density_samples)
    for label, name in label_names.items():
        points_to_plot = points_to_eval_emb[labels == label]
        ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], label=name, marker='x')
    ax.legend()
    return fig, ax
    

def plot_2d_log_density(points, labels, density_model, resolution=100, levels=10, label_names=None):
    """ Makes a 2d log-density plot of a space. 
    
    Parameters:
    -----------
    points : torch.tensor, shape [N, 2]
        points to place as points in the density space.
    labels : torch.tensor, shape [N]
        Labels for the points to place (will be added to the legend).
    density_model : torch.nn.Module
        A model that evaluates the logit space density.
    resolution : int
        Resolution for the plot. Default: 100
    levels : int
        How many countour levels. Default: 10
    label_names : dict or None
        Names of the labels. If None is given, each label is rerred to by its number.
    
    Returns:
    --------
    plt.Figure
        The plot.
    plt.axis.Axes
        The axis of the plot.
    """
    points = points.cpu().numpy()
    labels = labels.cpu().numpy()

    if points.shape[1] != 2:
        print(f'Cant plot 2d density for logit space of dimension {points.shape[1]}')
        return None, None

    if label_names is None:
        label_names = {label : f'{label}' for label in np.unique(labels)}

    density_model = density_model.cpu()

    (xmin, ymin), (xmax, ymax) = points.min(axis=0), points.max(axis=0)
    mesh = np.array(np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))).reshape(2, -1).T
    mesh = torch.tensor(mesh)
    mesh_log_density = np.log(density_model(mesh).cpu().numpy().T.reshape(resolution, resolution))
    xx, yy = mesh.cpu().numpy().T.reshape(2, resolution, resolution)
    fig, ax = plt.subplots(1, 1)
    ax.contourf(xx, yy, mesh_log_density, levels=20)
    for label, name in label_names.items():
        ax.scatter(points[:, 0][labels==label], points[:, 1][labels==label], c=_tableaeu_colors[label], label=name, marker='x', linewidth=1.0)
    ax.legend()
    return fig, ax

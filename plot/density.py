import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.stats import binned_statistic_2d
from openTSNE import TSNE
import umap
from util import is_outlier

_tableaeu_colors = list(mcolors.TABLEAU_COLORS.keys())


def plot_density(points_to_fit, points_to_eval, density_model, labels, label_names, 
                seed=1337, bins=20, levels=20, dimensionality_reduction='umap', num_samples=50000,
            sampling_stragey = 'random'):
    """ Creates a density plot for high dimensional points by using dimensionality reduction. 
    Also visualizes the density by creating a meshgrid in 2d space
    and using the inverse transform to find densities for these points or sampling.
    
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
    dimensionality_reduction : 'pca' or 'tsne' or 'umap'
        How to reduce the dimensionality of the points.
    sampling_strategy : 'random', 'convex_combinations', 'normal'
        How to sample points to find densities over a meshgrid in non invertible dimensionality reductions.

    Returns:
    --------
    plt.Figure
        The plot.
    plt.axis.Axes
        The axis of the plot.
    """
    
    bins_x, bins_y = bins, bins
    dimensionality_reduction = dimensionality_reduction.lower()
    sampling_stragey = sampling_stragey.lower()
    points_to_fit = points_to_fit.cpu()
    points_to_eval = points_to_eval.cpu()
    points = torch.cat([points_to_fit, points_to_eval], 0)
    
    
    if dimensionality_reduction == 'tsne':
        proj = TSNE(random_state = seed).fit(points.numpy())
    elif dimensionality_reduction == 'pca':
        proj = PCA(n_components=2, random_state=seed)
        proj.fit(points.numpy())
    elif dimensionality_reduction == 'umap':
        proj = umap.UMAP(random_state = seed)
        proj.fit(points.numpy())
    else:
        raise RuntimeError(f'Unsupported dimensionality reduction {dimensionality_reduction}')
    
    emb_to_fit = proj.transform(points_to_fit)
    emb_to_eval = proj.transform(points_to_eval)
    emb = np.concatenate([emb_to_fit, emb_to_eval], 0)
    
    if hasattr(proj, 'inverse_transform'):
        # Use the inverse transform of the projection
        mins, maxs = emb.min(0), emb.max(0)
        mins, maxs = mins - 0.1 * (maxs - mins), maxs + 0.05 * (maxs - mins) # Gives a margin for the density map
        xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], bins_x), np.linspace(mins[1], maxs[1], bins_y), indexing='ij')
        xx, yy = xx.reshape((-1, 1)), yy.reshape((-1, 1))
        grid = np.concatenate((xx, yy), axis=-1)
        
        density_grid = density_model(torch.tensor(proj.inverse_transform(grid)).float()).cpu().numpy()
        xx, yy, density_grid = xx.reshape((bins_x, bins_y)), yy.reshape((bins_x, bins_y)), density_grid.reshape((bins_x, bins_y))
    else:
        # Sample points within the box of the original data
        mins, maxs = points.min(0)[0], points.max(0)[0]
        mins, maxs = mins - 0.1 * (maxs - mins), maxs + 0.1 * (maxs - mins) # Gives a margin for the density map
        if sampling_stragey == 'random':
            samples = torch.rand(num_samples, points.size(1))
            samples *= maxs - mins
            samples += mins
        elif sampling_stragey == 'convex_combinations':
            k = 4 # This many points are in each convex combination
            coefs_mask = torch.zeros(num_samples, emb.shape[0])
            selected = torch.argsort(torch.randn(*coefs_mask.size()), 1)[:, :k]
            for row in range(selected.size(0)):
                coefs_mask[row][selected[row].tolist()] = 1.0
            
            coefs = torch.rand(*coefs_mask.size()) * coefs_mask
            coefs /= 0.25 * coefs.sum(1)[:, None]
            samples = coefs @ points
        elif sampling_stragey == 'normal':
            samples = torch.randn(num_samples, points.size(1))
        
        else:
            raise RuntimeError(f'Unknown sampling strategy for high dimesional space {sampling_stragey}')
        
        samples = torch.cat([samples, points])
        
        density_samples = density_model(samples).cpu().numpy()
        samples_emb = proj.transform(samples.numpy())
        
        density_grid, bins_x, bins_y, _ = binned_statistic_2d(
        samples_emb[:, 0], samples_emb[:, 1], density_samples, bins=(bins, bins))
        density_grid = density_grid.T
        bins_x, bins_y = 0.5 * (bins_x[1:] + bins_x[:-1]), 0.5 * (bins_y[1:] + bins_y[:-1])
        xx, yy = np.meshgrid(bins_x, bins_y)
    
    
    fig, ax = plt.subplots(1, 1)
    c = ax.contourf(xx, yy, density_grid, cmap='Reds', levels=levels)
    fig.colorbar(c, ax=ax)
    #ax.scatter(samples_emb[:, 0], samples_emb[:, 1], c=density_samples)
    for label, name in label_names.items():
        points_to_plot = emb_to_eval[labels == label]
        ax.scatter(points_to_plot[:, 0], points_to_plot[:, 1], label=name, marker='x')
    ax.scatter(emb_to_fit[:, 0], emb_to_fit[:, 1], label='Fit', marker='1')
    
    emb_mins, emb_maxs = emb.min(0), emb.max(0)
    emb_mins, emb_maxs = emb_mins - 0.05 * (emb_maxs - emb_mins), emb_maxs + 0.05 * (emb_maxs - emb_mins)
    
    ax.set_xlim(left = emb_mins[0], right = emb_maxs[0])
    ax.set_ylim(bottom = emb_mins[1], top = emb_maxs[1])
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

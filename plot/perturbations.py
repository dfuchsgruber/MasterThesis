import numpy as np
import matplotlib.pyplot as plt
from evaluation.lipschitz import local_lipschitz_bounds

def local_perturbations_plot(perturbations):
    """ Makes a plot for perturbations. 
    
    Parameters:
    -----------
    perturbations : dict
        Mapping from perturbation_magnitude -> tensor of output perturbations.

    Returns:
    --------
    fig : plt.Figure
        The figure of the plot.
    ax : plt.Axes
        The axis of the plot.
    slope_ys : float
        The slope of the linear function fit to median perturbations.
    slope_bound : float
        The slope of the linear function that bounds all perturbations. This is an empirical (tight) bound on the local Lipschitz constant.
    """
    xs, ys, ylower, yupper = [], [], [], []
    for eps, perturbs in perturbations.items():
        xs.append(eps)
        ys.append(perturbs.median().item())
        ylower.append(perturbs.min().item())
        yupper.append(perturbs.max().item())
        
    xs, ys, ylower, yupper = np.array(xs), np.array(ys), np.array(ylower), np.array(yupper)
    _, slope_ys, slope_yupper, _ = local_lipschitz_bounds(perturbations)
    
    fig, axs = plt.subplots(1, 1)
    axs.plot(np.array(xs), np.array(ys))
    axs.fill_between(np.array(xs), np.array(ylower), np.array(yupper), alpha=0.1)
    axs.plot(np.array([xs.min(), xs.max()]), slope_ys * np.array([xs.min(), xs.max()]), '--', 
             color='red', alpha=0.5, label=f'Average Median local perturbation (Slope={slope_ys:.2f})')
    axs.plot(np.array([xs.min(), xs.max()]), slope_yupper * np.array([xs.min(), xs.max()]), '--', 
             color='orange', alpha=0.5, label=f'Bound on local perturbation (Slope={slope_yupper:.2f})')
    
    axs.legend()
    axs.set_xlabel('Input perturbation')
    axs.set_ylabel('Output perturbation')
    axs.set_ylim(0, np.array(yupper).max())
    
    return fig, axs, slope_ys, slope_yupper
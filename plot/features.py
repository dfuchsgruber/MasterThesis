import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

_tableaeu_colors = list(mcolors.TABLEAU_COLORS.keys())

def plot_2d_features(points, labels, label_names=None):
    """ Makes a 2d log-density plot of a space. 
    
    Parameters:
    -----------
    points : torch.tensor, shape [N, 2]
        points to place as points in the density space.
    labels : torch.tensor, shape [N]
        Labels for the points to place (will be added to the legend).
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
        print(f'Cant plot 2d feature space of dimension {points.shape[1]}')
        return None, None

    if label_names is None:
        label_names = {label : f'{label}' for label in np.unique(labels)}
        
    fig, ax = plt.subplots(1, 1)
    for label, name in label_names.items():
        ax.scatter(points[:, 0][labels==label], points[:, 1][labels==label], c=_tableaeu_colors[label], label=name, marker='x', linewidth=1.0)
    ax.legend()
    return fig, ax

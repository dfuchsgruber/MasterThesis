import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_norms(norm, y, y_labels=None, norm_label='Norm', plot_labels=False, **kwargs):
    """ Plots the norm of instances by their label.
    
    Parameters:
    -----------
    norm : ndarray, shape [N]
        Norm of the instances.
    y : ndarray, shape [N]
        Labels for instances.
    y_labels : dict or None
        Names for the labels.
    norm_label : str
        Name to put for the norm (y axis)
    plot_labels : bool
        Whether to put the labels at the x axis.

    Returns:
    --------
    fig : plt.Figure
        The figure
    ax : plt.Axes
        The axis of the figure.
    """
    if y_labels is None:
        y_labels = {i : f'Class {i}' for i in np.unique(y)}
    fig, ax = plt.subplots(**kwargs)
    data = pd.DataFrame({
        'Label' : [y_labels[i] for i in y],
        norm_label : norm,
    })
    sns.stripplot(data = data, ax=ax, x='Label', y=norm_label, alpha=0.1, size=5, color='blue')
    sns.boxplot(data = data, ax=ax, x='Label', y=norm_label, 
                **{
                    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
                    'medianprops':{'color':'black'},
                    'whiskerprops':{'color':'black'},
                    'capprops':{'color':'black'}
    })
    if not plot_labels:
        ax.set_xticks(())
    return fig, ax

def plot_logit_cosine_angles(cos_angles, y, y_labels=None, class_labels=None, rows=None, cols=3, plot_labels=False, **kwargs):
    """ Plots the cosine of angles between features and logit weights as boxplots. 
    
    Parameters:
    -----------
    cos_angles : ndarray, shape [N, num_classes]
        Angle between each vertex feature and the corresponding logit weight.
    y : ndarray, shape [N]
        Labels for each vertex
    y_labels : dict or None
        Names for the labels.
    class_labels : dict or None
        Names for classes. If None, `y_labels` will be used.
    rows : int or None
        How many rows. If None is given, it will be infered from `cos_angles.shape` and `cols`
    cols : int
        How many cols.
    plot_labels : bool
        Whether to put the labels at the x axis.

    Returns:
    --------
    fig : plt.Figure
        Figure instance.
    axs : ndarray of plt.Axes
        Axes of the plot.
    """
    if y_labels is None:
        y_labels = {i : f'Class {i}' for i in np.unique(y)}
    if class_labels is None:
        class_labels = y_labels
    if rows is None:
        rows = int(np.around((cos_angles.shape[1] / cols) + .5))
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, **kwargs)
    for label, ax in enumerate(axs.flatten()):
        if label >= cos_angles.shape[1]:
            pass
        else:
            ax.set_title(f'Weights {class_labels[label]}')
            data = pd.DataFrame({
                'Label' : [y_labels[i] for i in y],
                'Cosine' : cos_angles[:, label],
            })
            sns.stripplot(data = data, ax=ax, x='Label', y='Cosine', alpha=0.1, size=5, color='blue')
            sns.boxplot(data = data, ax=ax, x='Label', y='Cosine', 
                        **{
                            'boxprops':{'facecolor':'none', 'edgecolor':'black'},
                            'medianprops':{'color':'black'},
                            'whiskerprops':{'color':'black'},
                            'capprops':{'color':'black'}
            })
            if not plot_labels:
                ax.set_xticks(())
    
    return fig, axs
    
    
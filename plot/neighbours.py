import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns

def plot_against_fraction_id_nbs(fraction_id_nbs, y, y_label='Proxy', y_log_scale=False, k_min=0, kind='box', bins=10):
    """ Plots some variable against the fraction of id neighbours in different k-hop neighbourhoods.
    
    Parameters:
    -----------
    fraction_id_nbs : torch.Tensor, shape [N, k]
        The fraction of id neighbours in every k hop neighbourhood to plot
    y : torch.Tensor, shape [N]
        The variable to plot
    y_label : str
        How the variable is called.
    y_log_scale : bool
        If to use a logarithmic scale for the y axis.
    k_min : int
        The first k-hop neighbourhood to consider.
    kind : 'box' or 'scatter'
        If to plot them as scatter plot (each vertex individually) or bin them and do a box plot.
    bins : int
        If kind = 'box', how many bins to use. That is, bins with width 1.0 / bins will be created
        and centered, such that 1.0 and 0.0 are their own bins (so technically its `bins + 1` bins.)
    
    Returns:
    --------
    fig : plt.Figure
        The figure
    axs : List[plt.Axes]
        The last axes of the facet grid.
    """
    if kind == 'scatter':
        df = pd.DataFrame({
            **{
                y_label : y.cpu().numpy()
            }, 
            **{
                f'{k}' : fraction_id_nbs[:, k].cpu().numpy() for k in range(k_min, fraction_id_nbs.size(1))
            }
        })
        df = df.melt(id_vars=[y_label], var_name='Neighbourhood', value_name='Fraction of in distirubtion neighbours')
        g = sns.relplot(data=df, y=y_label, x='Fraction of in distirubtion neighbours', col='Neighbourhood', alpha=0.5, kind='scatter')
    elif kind == 'box':
        bin_width = 1.0 / (bins)
        bin_edges = np.linspace(-bin_width * 0.5, 1 + bin_width * 0.5, bins + 2)
        bin_centers = np.abs(np.round((bin_edges[:-1] + bin_edges[1:]) * 0.5, 2))
        bin_idx = np.digitize(fraction_id_nbs.cpu().numpy(), bins=bin_edges) - 1
        df = pd.DataFrame({
            **{
                y_label : y,  
            },
            **{
                f'{k}' : bin_centers[bin_idx[:, k]] for k in range(k_min, fraction_id_nbs.size(1))
            },
        })
        df = df.melt(id_vars=[y_label], var_name='Neighbourhood', value_name='Fraction of in distirubtion neighbours')
        g = sns.catplot(data=df, y=y_label, x='Fraction of in distirubtion neighbours', col='Neighbourhood', kind='box',
            col_wrap = 2,
            **{
            'boxprops':{'facecolor':'none', 'edgecolor':'black'},
            'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},
            'capprops':{'color':'black'}})

        g.map_dataframe(sns.stripplot, y=y_label, x='Fraction of in distirubtion neighbours', alpha=0.1, size=5, color='blue')
    else:
        raise ValueError(f'Unsupported kind for neighbour plots {kind}')

    g.set_titles(col_template='{col_name}-hop neighbourhood')
    if y_log_scale:
        g.set(yscale="log")
    return g.fig, g.axes.flatten()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns

def plot_against_neighbourhood(xs, y, is_in_distribution, x_label='Quantity', y_label='Proxy', y_log_scale=False, k_min=0, kind='box', bins=10, x_min=None, x_max=None):
    """ Plots some variable against a quantity in different k-hop neighbourhoods.
    
    Parameters:
    -----------
    xs : torch.Tensor, shape [N, k]
        The quantity to plot against in different k-hop neighbourhoods.
    y : torch.Tensor, shape [N]
        The variable to plot
    is_in_distribution : torch.Tensor, shape [N]
        If a vertex is in distribution.
    x_label : str
        The name of the quantity that is plotted against.
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
    x_min : float, optional, default: None
        If given, the lower end of the binning space.
    x_max : float, optional, default: None
        If given, the upper end of the binning space.
    
    Returns:
    --------
    fig : plt.Figure
        The figure
    axs : List[plt.Axes]
        The last axes of the facet grid.
    """
    xs = xs.cpu().numpy()
    y = y.cpu().numpy()
    is_in_distribution = is_in_distribution.cpu().numpy()

    if kind == 'scatter':
        df = pd.DataFrame({
            **{
                y_label : y
            }, 
            **{
                f'{k}' : xs[:, k] for k in range(k_min, xs.shape[1])
            },
            **{
                'Tag' : ['In Distribution' if i else 'Out of Distribution' for i in is_in_distribution],
            },
        })
        df = df.melt(id_vars=[y_label, 'Tag'], var_name='Neighbourhood', value_name=x_label)
        g = sns.relplot(data=df, y=y_label, x=x_label, row='Tag', col='Neighbourhood', alpha=0.5, kind='scatter')
    elif kind == 'box':

        if x_min is None:
            x_min = xs.min(0)
        elif not isinstance(x_min, np.ndarray):
            x_min = np.array([x_min for _ in range(xs.shape[1])])
        if x_max is None:
            x_max = xs.max(0)
        elif not isinstance(x_max, np.ndarray):
            x_max = np.array([x_max for _ in range(xs.shape[1])])

        bin_widths = (x_max - x_min) / (bins)
        bin_centers, bin_idxs = [], []
        for k, bin_width in enumerate(bin_widths):
        
            bin_edges = np.linspace(x_min[k] - bin_width * 0.5, x_max[k] + bin_width * 0.5, bins + 2)
            bin_centers.append(np.abs(np.round((bin_edges[:-1] + bin_edges[1:]) * 0.5, 2)).astype(xs.dtype))

            bin_idx = np.abs(xs[:, k][:, None] - bin_centers[-1][None, :]).argmin(1)
            bin_idxs.append(bin_idx)
            
        df = pd.DataFrame({
            **{
                y_label : y,  
            },
            **{
                f'{k}' : bin_centers[k][bin_idxs[k]] for k in range(k_min, xs.shape[1])
            },
            **{
                'Tag' : ['In Distribution' if i else 'Out of Distribution' for i in is_in_distribution],
            },
        })
        # Do *NOT* use sharex=True, as it will not work...
        # each call of map_dataframe will ruin the shared x-axis of previous plots...
        df = df.melt(id_vars=[y_label, 'Tag'], var_name='Neighbourhood', value_name=x_label)
        g = sns.FacetGrid(data=df, row='Tag', col = 'Neighbourhood', sharex=False, height=5, margin_titles=True)
        g.map_dataframe(sns.boxplot, x_label, y_label, **{
                    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
                    'medianprops':{'color':'black'},
                    'whiskerprops':{'color':'black'},
                    'capprops':{'color':'black'}})
        g.map_dataframe(sns.stripplot, x_label, y_label, alpha=0.1, size=5, color='blue')
    else:
        raise ValueError(f'Unsupported kind for neighbour plots {kind}')

    g.set_titles(col_template='{col_name}-hop neighbourhood', row_template='{row_name}')
    if y_log_scale:
        g.set(yscale="log")
    return g.fig, g.axes.flatten()
from contextlib import contextmanager
import os, sys
import torch
import numpy as np
import networkx as nx
from collections import Mapping
import scipy.sparse as sp
from typing import Dict, Any, Iterable

@contextmanager
def suppress_stdout(supress=True):
    """ From: https://stackoverflow.com/a/25061573 """
    if supress:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:  
                yield
            finally:
                sys.stdout = old_stdout
    else:
        try:
            yield
        finally:
            pass

def is_outlier(array, quantile=0.95):
    """ Detects outliers as samples that are not within a certain quantile of the data. 
    
    Parameters:
    -----------
    array : ndarray, shape [N]
        The array to find outliers in.
    quantile : float
        How much of the data to include.

    Returns:
    --------
    outliers : ndarray, shape [N]
        Array that masks all outliers, i.e. `outliers[i] == True` if a point is identified as outlier.    
    """
    finite = np.isfinite(array)
    array_finite = array[finite]
    idxs = np.argsort(array_finite)
    delta = (1 - quantile) / 2
    upper, lower = int((quantile + delta) * array_finite.shape[0]), int(delta * array_finite.shape[0])
    idxs = idxs[lower:upper]
    # Translate to the whole array
    idxs = np.arange(array.shape[0])[finite][idxs]
    is_outlier = np.ones_like(array, dtype=bool)
    is_outlier[idxs] = False
    return is_outlier

def random_derangement(N, rng=None):
    """ Returns a random permutation of [0, 1, ..., N] that has no fixed points (derangement).
    
    Parameters:
    -----------
    N : int
        The number of elements to derange.
    rng : np.random.RandomState or None
        The random state to use. If None is given, a random RandomState is generated.

    References:
    -----------
    Taken from: https://stackoverflow.com/a/52614780
    """
    if rng is None:
        rng = np.random.RandomState(np.random.randint(1 << 32))
    
    # Edge cases
    if N == 0:
        return np.array([], dtype=int)
    elif N == 1:
        return np.array([0], dtype=int)

    original = np.arange(N)
    new = rng.permutation(N)
    same = np.where(original == new)[0]
    _iter = 0
    while len(same) != 0:
        if _iter + 1 % 100 == 0:
            print('iteration', _iter)
        swap = same[rng.permutation(len(same))]
        new[same] = new[swap]
        same = np.where(original == new)[0]
        if len(same) == 1:
            swap = rng.randint(0, N)
            new[[same[0], swap]] = new[[swap, same[0]]]
    return new

def format_name(name_fmt, args, config, delimiter=':'):
    """ Formats a name using arguments from a config. That is, if '{i}' appears in
    `name_fmt`, it is replaced with the `i`-th element in `args`. Each element in `args`
    is a path in the config dict, where levels are separated by '.'.

    Example:
    `format_name('name-{0}-{1}', args = ['foo', 'level.sub'], config = {'foo' : 'bar', 'level' : {'sub' : [1, 2]}})`
    returns
    `name-bar-[1-2]

    
    Parameters:
    -----------
    name_fmt : str
        The format string.
    args : list
        Paths to each argument for the config string.
    config : dict
        A nested configuration dict.
    delimiter : str
        The delimiter to access different levels of the config dict with paths defined in `args`.

    Returns:
    --------
    formatted : str
        The formated name.
    """
    parsed_args = []
    for arg in args:
        path = arg.split(delimiter)
        arg = config
        for x in path:
            arg = arg[x]
        if isinstance(arg, list):
            arg = '[' + '-'.join(map(str, arg)) + ']'
        elif isinstance(arg, bool):
            arg = str(arg)[0].upper()
        parsed_args.append(str(arg))
    return name_fmt.format(*parsed_args)


def get_k_hop_neighbourhood(edge_list, k_max, k_min = None):
    """ Gets all vertices in the k-hop neighbourhood of vertices. 
    
    Parameters:
    -----------
    edge_list : torch.Tensor, shape [2, num_egdes]
        The graph structure.
    k_max : int
        Vertices returned can be at most `k_max` hops away from a source.
    k_min : int or None
        Vertices returned have to be at least `k_min``hops away from a source. 
        If `None`, `k_min` is set equal to `k_max`, which corresponds to the k-hop
        neighbourhoods exactly.
    smaller_or_equal : bool
        If True, the <= k-hop neighbourhood is returned (vertices AT MOST k hops away).
    
    Returns:
    --------
    k-hop-neighbourhood : dict
        A mapping from vertex_idx to a tuple of vertex_idxs in the k-hop neighbourhood.
    """
    if k_min is None:
        k_min = k_max
    G = nx.Graph()
    G.add_edges_from(edge_list.numpy().T)
    return {
        src : tuple(n for n, path in nx.single_source_shortest_path(G, src, cutoff=k_max).items() if len(path) <= k_max + 1 and len(path) >= k_min + 1) 
        for src in G.nodes
    }

def dict_to_tuple(d):
    """ Creates a nested tuple from a dict (to make it immutable and hashable) 
    
    Parameters:
    -----------
    d : dict
        The dict to convert

    Returns:
    --------
    t : tuple
        The deep tuple representation of `d`.
    """
    elements = []
    for k, v in d.items():
        if isinstance(v, Mapping):
            v = dict_to_tuple(v)
        elif isinstace(v, Iterable):
            v = tuple(dict_to_tuple(e) for e in v)
        elements.append((k, v))
    return tuple(elements)


def calibration_curve(probs, y_true, bins=10, eps=1e-12):
    """ Calculates the calibration curve for predictions.
    
    Parameters:
    -----------
    probs : torch.Tensor, shape [n, num_classes]
        Predicted probabilities.
    y_true : torch.Tensor, shape [n]
        True class labels.
    bins : int
        The number of bins to use.
    eps : float 
        Epsilon to prevent division by zero.
    
    Returns:
    --------
    bin_edges : ndarray, shape [bins + 1]
        Edges for the bins
    bin_confidence : ndarray, shape [bins]
        Average confidence per bin
    bin_accuracy : ndarray, shape [bins]
        Average accuracy per bin
    bin_weight : ndarray, shape [bins]
        Bin weights (i.e. fraction of samples that is in each bin)
    """
    n, c = probs.size()
    max_prob, hard = probs.detach().cpu().max(dim=-1)
    y_true_one_hot = np.eye(c)[y_true.detach().cpu().numpy()]
    
    bin_edges = np.linspace(0., 1., bins + 1)
    bin_width = 1 / bins
    digitized = np.digitize(max_prob.numpy(), bin_edges)
    digitized = np.maximum(np.minimum(digitized, bins), 1) - 1 # Push values outside the bins into the rightmost and leftmost bins
    
    bins_sum = np.bincount(digitized, minlength=bins, weights=max_prob.numpy())
    bins_size = np.bincount(digitized, minlength=bins)
    is_correct = y_true_one_hot[range(n), hard]
    
    bin_confidence = bins_sum / (bins_size + eps)
    bin_accuracy = np.bincount(digitized, minlength=bins, weights=is_correct) / (bins_size + eps)
    bin_weight = bins_size / bins_size.sum()
    
    return bin_edges, bin_confidence, bin_accuracy, bin_weight

def sparse_max(A, B):
    """
    Return the element-wise maximum of sparse matrices `A` and `B`.

    References:
    -----------
    Taken from: https://stackoverflow.com/questions/19311353/element-wise-maximum-of-two-sparse-matrices
    """
    AgtB = (A > B).astype(int)
    M = AgtB.multiply(A - B) + B
    return M

def get_cache_path():
    """ Returns the path to the standard cache.
    
    Returns:
    --------
    cache_path : str
        Path the standard chache.
    """
    xdg_cache_home = os.environ.get('XDG_CACHE_HOME') or \
           os.path.join(os.path.expanduser('~'), '.cache')

    return os.path.join(xdg_cache_home, 'MasterThesis')

def invert_mapping(d: Mapping) -> dict:
    """ Inverts a dictionary. 
    
    Parameters:
    -----------
    d : dict
        The dict to invert.
    
    Returns:
    --------
    inv : dict
        The inverted dict.
    """
    inv = {}
    for k, v in d.items():
        if v in inv:
            raise ValueError(f'Can not invert dict because two keys map to value {v}')
        inv[v] = k
    return inv

def approximate_page_rank_matrix(edge_index: np.ndarray, num_vertices: int, diffusion_iterations: int = 16, 
        alpha: float = 0.2, edge_weights=None, add_self_loops: bool = False):
    """ Calculates the approximate page rank matrix for a given graph. 
    
    Parameters:
    -----------
    edge_index : ndarray, shape [2, E]
        Edge indices.
    num_vertices : int
        Number of vertices in the graph.
    diffusion_iterations : int, default: 16
        How many diffusion iterations to perform
    alpha : float, default: 0.2
        Teleportation probability (higher means more focus on a vertex itself)
    edge_weights : ndarray, shape [E], optional
        Edge weights. If not given, ones are used.
    add_self_loops : bool, deault: False
        If self loops should be added.

    Returns:
    --------
    pi : ndarray, shape [N, N]
        Approximate page rank matrix.
    """
    if edge_weights is None:
        edge_weights = np.ones(edge_index.shape[1])
    A = sp.coo_matrix((edge_weights, edge_index), shape=(num_vertices, num_vertices)).tocsr()

    # Normalize adjacency
    if add_self_loops:
        A += sp.coo_matrix(np.eye(A.shape[0]))
    degrees = A.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A = D.dot(A).dot(D)

    ppr = np.ones((num_vertices, num_vertices)) / num_vertices
    for _ in range(diffusion_iterations):
        ppr = (alpha * np.eye(num_vertices)) + ((1 - alpha) * (A @ ppr))
    return ppr

def aggregate_matching(flags: Dict[Any, bool]) -> Any:
    """ If all values in the input dict match, it returns that value. Otherwise it raises an `RuntimeExpection`. """
    if len(set(flags.values())) != 1:
        raise RuntimeError(f'Not matching values in {flags}')
    else:
        return list(flags.values[0])

def all_equal(x: Iterable) -> bool:
    """ Checks if all elements in an iterable are the same"""
    x = list(x)
    return all(i == x[0] for i in x)
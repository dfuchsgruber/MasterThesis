import numpy as np
import scipy.sparse as sp

GLOBAL_SPLIT_SEED = 1337

def stratified_split(ys, num_splits, sizes=[0.05, 0.15, 0.8], random_state=GLOBAL_SPLIT_SEED):
    """ Performs several splits of a dataset into index sets using a stratified strategy.
    
    Parameters:
    -----------
    ys : ndarray, shape [N]
        Labels that are used for a stratified split.
    num_splits : int
        How many splits are calculated.
    sizes : sequence of ints
        Size of each split. Should sum up to 1.0
    random_state : int or None
        If given, seed for generating all splits to ensure reproducability.

    Returns:
    --------
    mask : list or ndarrays, shape [sizes, num_splits, N]
        Masks for all sets.
    """
    sizes = np.array(sizes)
    assert np.allclose(sizes.sum(), 1.0), f'Sizes should sum up to 1.0'

    if random_state is not None:
        np.random.seed(random_state)
        
    mask = np.zeros((len(sizes), num_splits, len(ys)), dtype=bool)
    for split_idx in range(num_splits):
        for v, count in zip(*np.unique(ys, return_counts=True)):
            idxs = np.where(ys == v)[0]
            np.random.shuffle(idxs)
            
            # Partition this subset by splitting the shuffled idx in portions
            start_idx = 0
            sizecumsum = np.cumsum(sizes)
            sizecumsum[-1] = 1.0 # A bit hacky, but avoids numerical issues
            for set_idx, endpoint in enumerate(sizecumsum):
                end_idx = int(len(idxs) * endpoint)
                mask[set_idx, split_idx][idxs[start_idx : end_idx]] = True
                start_idx = end_idx

    return mask

def data_get_num_attributes(data):
    return data[0].x.size(1)

def data_get_num_classes(data):
    return int((data[0].y.max() + 1).item())

def graph_make_symmetric(edge_index):
    """ Makes a graph symmetric. 
    
    Parameters:
    -----------
    edge_index : ndarray, shape [2, E]
        Endpoints of edges.
    
    Returns:
    --------
    edge_index : ndarray, shape [2, E']
        Endpoints of edges in the symmetric graph.
    """
    n = edge_index.max() + 1
    A = sp.coo_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(n, n))
    A = A + A.T
    return np.array(A.nonzero())

def graph_get_largest_connected_component(x, edge_index, y, vertex_to_idx):
    """ Selects the largest connected component in the graph.

    Parameters:
    -----------
    x : ndarray, shape [N, D]
        Attribute matrix.
    edge_index : ndarray, shape [2, E]
        Endpoints of edges.
    y : ndarray, shape [N]
        Labels.
    vertex_to_idx : dict
        A mapping from vertex_id -> idx in the graph.
    make_symmetric : bool
        If the graph is made symmetric before normalization.
    min_class_ratio : float
        If a portion of vertices of a class is less than this value, it will be removed.

    Returns:
    --------
    x : ndarray, shape [N', D]
        New attribute matrix.
    edge_index : ndarray, shape [2, E']
        New endpoints of edges.
    y : ndarray, shape [N']
        New labels.
    vertex_to_idx : dict
        New mapping from vertex_id -> idx in the graph.
    """
    n = x.shape[0]
    A = sp.coo_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(n, n))
    n_components, labels = sp.csgraph.connected_components(A)
    label_names, label_counts = np.unique(labels, return_counts=True)
    label_lcc = label_names[label_counts.argmax()]
    return graph_select_idxs(labels == label_lcc, x, edge_index, y, vertex_to_idx)

def graph_normalize(x, edge_index, y, vertex_to_idx, label_to_idx, make_symmetric=True, min_samples_per_class=20 / 0.05, min_class_count=0.0, verbose=True):
    """ Normalizes a graph by repeating the following steps to a fixed point:

    1. Select largest connected component
    2. Remove classes that are underrepresented
    
    Parameters:
    -----------
    mask : ndarray, shape [N]
        Mask for vertices to select. If mask[i] == `False`, the vertex will be discarded.
    x : ndarray, shape [N, D]
        Attribute matrix.
    edge_index : ndarray, shape [2, E]
        Endpoints of edges.
    y : ndarray, shape [N]
        Labels.
    vertex_to_idx : dict
        A mapping from vertex_id -> idx in the graph.
    label_to_idx : dict
        A mapping from label_id -> label idx.
    make_symmetric : bool
        If the graph is made symmetric before normalization.
    min_class_count : float
        If a portion of vertices of a class is less than this value, it will be removed. Setting this parameter to 0.0 effectively ignores it alltogether.
        Setting it to k / p ensures that if a portion of the data with size p*n is sampled stratifyingly, at least k samples of this class are retained.
    verbose : bool
        If True, normalization steps are printed.

    Returns:
    --------
    x : ndarray, shape [N', D]
        New attribute matrix.
    edge_index : ndarray, shape [2, E']
        New endpoints of edges.
    y : ndarray, shape [N']
        New labels.
    vertex_to_idx : dict
        New mapping from vertex_id -> idx in the graph.
    label_to_idx : dict
        New mapping from label_id -> label idx.
    """
    if make_symmetric:
        edge_index = graph_make_symmetric(edge_index)
    n = x.shape[0]
    idx_to_label = {idx : label for label, idx in label_to_idx.items()} # For printing
    print(f'Normalizing graph with {x.shape[0]} vertices...')
    while n > 0:
        # Select largest connected component
        x, edge_index, y, vertex_to_idx = graph_get_largest_connected_component(x, edge_index, y, vertex_to_idx)
        if verbose:
            print(f'After selecting lcc, graph has {x.shape[0]} vertices.')
        # Remove underepresented classes
        mask = np.ones(x.shape[0], dtype=bool)
        for label, count in zip(*np.unique(y, return_counts=True)):
            if count < min_class_count:
                mask[y == label] = False
                if verbose:
                    print(f'Class {idx_to_label[label]} ({label}) is underrepresented ({count:.2f} < {min_class_count:.2f}), hence it is removed.')
        x, edge_index, y, vertex_to_idx = graph_select_idxs(mask, x, edge_index, y, vertex_to_idx)
        if verbose:
            print(f'After removing underrepresented classes, graph has {x.shape[0]} vertices.')
        if x.shape[0] == n:
            break # Fixed point
        else:
            n = x.shape[0] # Continue

    # Fix label_to_idx
    label_masks = {label : y == idx for label, idx in label_to_idx.items() if idx in y}
    label_to_idx = {label : idx for idx, label in enumerate(label_masks.keys())}
    y = -np.ones(y.shape, dtype=int)
    for label, mask in label_masks.items():
        y[mask] = label_to_idx[label]
    assert (y >= 0).all()
    
    return x, edge_index, y, vertex_to_idx, label_to_idx

def graph_select_idxs(mask, x, edge_index, y, vertex_to_idx):
    """ Reduces a graph by subsetting its vertices. 
    
    Parameters:
    -----------
    mask : ndarray, shape [N]
        Mask for vertices to select. If mask[i] == `False`, the vertex will be discarded.
    x : ndarray, shape [N, D]
        Attribute matrix.
    edge_index : ndarray, shape [2, E]
        Endpoints of edges.
    y : ndarray, shape [N]
        Labels.
    vertex_to_idx : dict
        A mapping from vertex_id -> idx in the graph.

    Returns:
    --------
    x : ndarray, shape [N', D]
        New attribute matrix.
    edge_index : ndarray, shape [2, E']
        New endpoints of edges.
    y : ndarray, shape [N']
        New labels.
    vertex_to_idx : dict
        New mapping from vertex_id -> idx in the graph.
    """
    N = x.shape[0]
    x = x[mask]
    y = y[mask]
    A = sp.coo_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(N, N))
    A = A.tocsr()[mask].tocsc()[:, mask]
    edge_index = np.array(A.nonzero())
    idx_mapping = {idx_old : idx for idx, idx_old in enumerate(np.arange(N, dtype=int)[mask])}
    vertex_to_idx = {v : idx_mapping[idx] for v, idx in vertex_to_idx.items() if idx in idx_mapping}
    return x, edge_index, y, vertex_to_idx


if __name__ == '__main__':
    ys = np.array([0] * 100 + [1] * 150 + [2] * 250)
    mask = stratified_split(ys, 4, sizes=[0.3, 0.3, 0.4])
    print(mask.shape)
    print((mask.sum(0) == 1).all())
    x = np.arange(20).reshape((5, 4))
    y = np.array([1, 0, 1, 1, 0])
    edge_index = np.array([[0, 1], [0, 0], [2, 1], [2, 3], [2, 2], [3, 1], [4, 4]]).T
    vertex_to_idx = {'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4}
    label_to_idx = {'class0' : 0, 'class1' : 1}
    mask = np.array([0, 1, 0, 1, 1], dtype=bool)
    print(graph_select_idxs(mask, x, edge_index, y, vertex_to_idx))
    x, edge_index, y, vertex_to_idx, label_to_idx = graph_normalize(x, edge_index, y, vertex_to_idx, label_to_idx, make_symmetric=True, min_class_ratio=0.3)
    print(x, edge_index.T, y, vertex_to_idx, label_to_idx)
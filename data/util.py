import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Dataset, Data
from data.base import SingleGraphDataset
import torch
from warnings import warn

from seed import data_split_seeds

def data_get_summary(dataset, prefix='\t'):
    """ Gets the summary string of a dataset. 
    
    Parameters:
    -----------
    dataset : torch_geometric.data.Dataset
        The dataset to summarize.
    prefix : str
        Prefix for each line.
    
    Returns:
    --------
    summary : str
        The summary.
    """
    data = dataset[0]
    summary = []
    summary.append(f'{prefix}Number of Vertices : {data.x.size()[0]}')
    summary.append(f'{prefix}Number of Vertices in Mask : {data.x[data.mask].size()[0]}')
    summary.append(f'{prefix}Number of Labels : {len(torch.unique(data.y))}')
    summary.append(f'{prefix}Number of Labels in Mask : {len(torch.unique(data.y[data.mask]))}')
    summary.append(f'{prefix}Number of Edges : {data.edge_index.size()[1]}')
    summary.append(f'{prefix}Label mapping : {data.label_to_idx}')
    return '\n'.join(summary)


def get_label_mask(labels, select_labels):
    """ Gets a mask that selects only elements in `select_labels`.
    
    Parameters:
    -----------
    labels : ndarray, shape [N]
        Labels to select from.
    select_labels : iterable
        Labels to mask.
    
    Returns:
    --------
    mask : ndarray, shape [N]
        A mask that selects only labels from `select_labels`, i.e. `mask[i] == True` if and only if `labels[i] in select_labels`.
    """
    return np.array([l in select_labels for l in labels], dtype=bool)

def sample_uniformly(labels, select_lables, num_samples, mask, rng=None):
    """ Uniformly samples vertices within a mask. 
    
    Parameters:
    -----------
    labels : ndarray, shape [N]
        Labels to sample from.
    select_labels : iterable
        All classes to draw samples from.
    num_samples : int
        How many samples to draw from each class.
    mask : ndarray, shape [N]
        Which vertices are available to the sampling.
    rng : np.random.RandomState or None
        A random state to ensure reproducability.
    
    Returns:
    --------
    sample_mask : ndarray, shape [N]
        Which vertices were sampled.
    """
    if rng is None:
        rng = np.random.RandomState()
    all_labels = set(select_lables)
    sample_mask = np.zeros_like(labels, dtype=bool)
    for label in all_labels:
        idxs = np.where(mask & (labels == label))[0]
        if idxs.shape[0] < num_samples:
            raise RuntimeError(f'Could not sample {num_samples} vertices from class {label}, only {idxs.shape[0]} present.')
        rng.shuffle(idxs)
        sample_mask[idxs[:num_samples]] = True
    assert sample_mask.sum() == num_samples * len(all_labels)

    return sample_mask

def uniform_split_with_fixed_test_set_portion(data, num_splits, num_train=20, num_val=20, portion_test_fixed=0.2, train_labels='all', train_labels_remove_other=False, 
        val_labels='all', val_labels_remove_other=False, compress_train_labels=True, compress_val_labels=True):
    """ Splits the data by uniformaly sampling a fixed amount of vertices per class for each dataset.
    The procedure goes as follows:
    A. Split into a fixed test-set and a non-fixed dataset in a stratified manner
    B. Each following split is obtained from the non-fixed dataset this way:
        (0.) Reduce graphs for training and validation lables
        1. Sample `num_train` vertices per training-class from the (reduced) training graph
        2. Sample `num_val` vertices per training-class from the (reduced) training graph for a validation set on the same graph as the training data for monitoring
        3. Sample `num_val` vertices per validation-class from the (reduced) validation graph for a global validation set after training
        4. All vertices not sampled so far make the test set

    Parameters:
    -----------
    data : torch_geometric.data.Data
        A base data sample that contains the graph on which everything is built upon.
    num_splits : int
        How many splits to generate.
    num_train : int
        How many vertices to sample per class for the (reduced) training graph.
    num_val : int
        How many vertices to sample per class for the (reduced) training and validation graph.
    portion_test_fixed : float
        Which portion of the data to fix as testing data that is not used ever.
    train_labels : iterable or 'all'
        Labels for the training graph.
    train_labels_remove_other : bool
        If `True`, vertices with a label not in `train_labels` will be removed from the graph entirely instead of being masked.
    val_labels : iterable or 'all'
        Labels for the validation graph.
    val_labels_remove_other : bool
        If `True`, vertices with a label not in `val_labels` will be removed from the graph entirely instead of being masked.
    compress_train_labels : bool
        If `True`, labels on the training graph are mapped onto (0, 1, ..., k)
    compress_val_labels : bool
        If `True`, labels on the validation graph are mapped onto (0, 1, ..., k)
    
    Returns:
    --------
    data_list : list
        A list of different data splits. Each element is 4-tuple of data_train, data_val, data_val_all_classes, data_test.
    data_test_fixed : torch_geometric.data.Dataset
        Fixed test dataset that is consistent among all splits.
    """
    seeds = data_split_seeds(num_splits + 1) # The first seed is used to fix the shared testing data
    
    # First stratified split to get the untouched test data
    mask_fixed, mask_non_fixed = stratified_split(data.y.numpy(), seeds[0:1], [portion_test_fixed, 1 - portion_test_fixed])[:, 0, :]

    all_labels = set(np.unique(data.y.numpy()))

    # Reduce the graph for training
    if train_labels == 'all':
        train_labels = all_labels.copy()
    else:
        train_labels = set(train_labels)
    if train_labels_remove_other:
        x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train_graph = graph_select_labels(data.x.numpy(), 
            data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, train_labels, connected=True, _compress_labels=compress_train_labels)
    else:
        x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train_graph = (data.x.numpy(), 
            data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx.copy(), data.label_to_idx.copy(), torch.ones_like(data.y).bool())

    # Reduce graph for validation
    if val_labels == 'all':
        val_labels = all_labels.copy()
    else:
        val_labels = set(val_labels)
    if val_labels_remove_other:
        x_val, edge_index_val, y_val, vertex_to_idx_val, label_to_idx_val, mask_val_graph = graph_select_labels(data.x.numpy(), 
            data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, val_labels, connected=True, _compress_labels=compress_val_labels)
    else:
        x_val, edge_index_val, y_val, vertex_to_idx_val, label_to_idx_val, mask_val_graph = (data.x.numpy(), 
            data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx.copy(), data.label_to_idx.copy(), torch.ones_like(data.y).bool())
    
    # Check if the validation graph is a superset of the training graph
    if not mask_val_graph[mask_train_graph].all():
        warn(f'Validation graph is not a superset of the training graph!')

    data_list = []
    # Sample `train_portion` vertices per class both for training and reduced validation sets
    for split_idx in range(num_splits):
        rng = np.random.RandomState(seeds[split_idx + 1])
        # Sample train idxs
        mask_train = sample_uniformly(data.y.numpy(), train_labels, num_train, mask_train_graph & mask_non_fixed, rng=rng)
        data_train = SingleGraphDataset(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train[mask_train_graph])
        # Sample val reduced idxs, disjunct from train idxs
        mask_val_reduced = sample_uniformly(data.y.numpy(), train_labels, num_val, mask_train_graph & (~mask_train) & mask_non_fixed, rng=rng)
        data_val_reduced = SingleGraphDataset(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_val_reduced[mask_train_graph])
        # Sample val idxs, disjunct from train idxs
        mask_val = sample_uniformly(data.y.numpy(), val_labels, num_val, mask_val_graph & (~mask_train) & mask_non_fixed, rng=rng)
        data_val = SingleGraphDataset(x_val, edge_index_val, y_val, vertex_to_idx_val, label_to_idx_val, mask_val[mask_val_graph])
        # Test idx are the non-used idxs (which are not fixed)
        mask_test = (~(mask_train | mask_val_reduced | mask_val)) & mask_non_fixed
        data_test = SingleGraphDataset(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, mask_test)
        
        # Sanity checks
        assert ((mask_train.astype(int) + mask_val_reduced.astype(int)) <= 1).all(), f'Training and reduced Validation data are not disjunct'
        assert ((mask_train.astype(int) + mask_val.astype(int)) <= 1).all(), f'Training and full Validation data are not disjunct'
        assert (((mask_train | mask_val | mask_val_reduced).astype(int) + mask_test.astype(int)) <= 1).all(), f'Training & Validation and Testing data are not disjunct'
        assert (((mask_train | mask_val | mask_val_reduced | mask_test).astype(int) + mask_fixed.astype(int)) <= 1).all(), f'Non-fixed and fixed data are not disjunct'

        for idx, dataset in enumerate((data_train, data_val, data_val_reduced, data_test)):
            assert (dataset[0].x.size()[0] == dataset[0].mask.size()[0])
            assert (dataset[0].y.size() == dataset[0].mask.size())
            assert (dataset[0].edge_index <= dataset[0].x.size()[0]).all()

        data_list.append((data_train, data_val_reduced, data_val, data_test))

    return data_list, SingleGraphDataset(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, mask_fixed)

def stratified_split_with_fixed_test_set_portion(ys, num_splits, portion_train=0.05, portion_val=0.15, portion_test_fixed=0.2, portion_test_not_fixed=0.6):
    """ Splits the dataset using a stratified strategy into training, validation and testing data.
    A certain portion of the testing data will be fixed and shared among all splits.

    Parameters:
    -----------
    ys : ndarray, shape [N]
        Labels that are used for a stratified split.
    num_splits : int
        How many splits are performed.
    portion_train : float
        Training portion.
    portion_val : float
        Validation portion.
    portion_test_fixed : float
        Testing portion that is shared among all splits.
    portion_test_not_fixed : float
        Testing portion that is not shared among all splits.
    
    Returns:
    --------
    mask : ndarray, shape [3, num_splits, N]
        Masks for these 3 splits: training, validation, testing_not_fixed
    mask_fixed : ndarray, shape [N]
        Mask for test data that is fixed among all splits.
    """
    seeds = data_split_seeds(num_splits + 1) # The first seed is used to fix the shared testing data
    assert np.allclose(portion_train + portion_val + portion_test_not_fixed + portion_test_fixed, 1.0), f'Dataset splits dont add to 1.0'
    norm = portion_train + portion_val + portion_test_not_fixed
    mask_non_fixed, mask_fixed = stratified_split(ys, seeds[0:1], [norm, 1 - norm])[:, 0, :]

    mask = np.zeros((3, num_splits, ys.shape[0]), bool)
    mask[:, :, mask_non_fixed] = stratified_split(ys[mask_non_fixed], seeds[1:], sizes=np.array([portion_train, portion_val, portion_test_not_fixed]) / norm)
    return mask, mask_fixed

def stratified_split(ys, seeds, sizes=[0.05, 0.15, 0.8]):
    """ Performs several splits of a dataset into index sets using a stratified strategy.
    
    Parameters:
    -----------
    ys : ndarray, shape [N]
        Labels that are used for a stratified split.
    seeds : int
        Seed for each split.
    sizes : sequence of ints
        Size of each split. Should sum up to 1.0

    Returns:
    --------
    mask : list or ndarrays, shape [len(sizes), len(seeds), N]
        Masks for all sets.
    """
    sizes = np.array(sizes)
    assert np.allclose(sizes.sum(), 1.0), f'Sizes should sum up to 1.0'

        
    mask = np.zeros((len(sizes), len(seeds), len(ys)), dtype=bool)
    for split_idx, seed in enumerate(seeds):
        for v, count in zip(*np.unique(ys, return_counts=True)):
            idxs = np.where(ys == v)[0]
            rng = np.random.RandomState(seed)
            rng.shuffle(idxs)
            
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
    return data.x.size(1)

def data_get_num_classes(data):
    return int((data.y[data.mask].max() + 1).item())

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
    mask : ndarray, shape [N]
        Mask for old vertices that were selected.
    """
    n = x.shape[0]
    A = sp.coo_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(n, n))
    n_components, labels = sp.csgraph.connected_components(A)
    label_names, label_counts = np.unique(labels, return_counts=True)
    label_lcc = label_names[label_counts.argmax()]
    x, edge_index, y, vertex_to_idx = graph_select_idxs(labels == label_lcc, x, edge_index, y, vertex_to_idx)
    return x, edge_index, y, vertex_to_idx, labels == label_lcc

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
        x, edge_index, y, vertex_to_idx, _ = graph_get_largest_connected_component(x, edge_index, y, vertex_to_idx)
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

def remap_labels(labels, label_to_idx, remapping):
    """ Remaps labels of a graph. 
    
    Parameters:
    -----------
    labels : ndarray, shape [N]
        Labels to remap.
    label_to_idx : dict
        A dict mapping a label name to its index.
    remapping : dict
        A mapping from label_old -> label_new.
    
    Returns:
    --------
    labels : ndarray, shape [N]
        Remapped labels.
    label_to_idx : dict
        Mapping from label name to new index.
    """
    labels_new = -np.ones_like(labels)
    label_to_idx_new = {}
    idx_to_label = {idx : label for label, idx in label_to_idx.items()}
    for label_old, label_new in remapping.items():
        labels_new[labels == label_old] = label_new
        label_to_idx_new[idx_to_label[label_old]] = label_new
    assert not (labels_new == -1).any()
    return labels_new, label_to_idx_new

def compress_labels(labels, label_to_idx):
    """ Compresses the labels of a graph. For examples the labels {0, 1, 3, 4} will be re-mapped
    to {0, 1, 2, 3}. Labels will be compressed in ascending order. 
    
    Parameters:
    -----------
    labels : ndarray, shape [N]
        Labels to compress.
    label_to_idx : dict
        Mapping from label name to its index.
    
    Returns:
    --------
    labels : ndarray, shape [N]
        Compressed labels.
    label_to_idx : dict
        Mapping from label name to compressed index.
    compression : dict
        Mapping from old index to new index.
    """
    label_set_sorted = np.sort(np.unique(labels))
    compression = {idx_old : idx_new for idx_new, idx_old in enumerate(label_set_sorted)}
    labels_new, label_to_idx_new = remap_labels(labels, label_to_idx, compression)
    return labels_new, label_to_idx_new, compression

def graph_select_labels(x, edge_index, y, vertex_to_idx, label_to_idx, select_labels, connected=True, _compress_labels=True):
    """ Gets a subgraph that only has vertices within a certain set of labels.
    
    Parameters:
    -----------
    x : ndarray, shape [N]
        Attribute matrix.
    edge_index : ndarray, shape [2, E]
        Edge endpoints.
    y : ndarray, shape [N]
        Labels.
    vertex_to_idx : dict
        Mapping from vertex name to its index in the arrays.
    label_to_idx : dict
        Mapping from label name to its value in `y`.
    select_labels : iterable
        Labels to select.
    connected : bool
        If `True`, only the largest connected component of the resulting subgraph is selected.
    _compress_labels : bool
        If `True`, labels are remapped to {0, 1, ... k}

    Returns:
    --------
    x' : ndarray, shape [N']
        New wttribute matrix.
    edge_index' : ndarray, shape [2, E']
        New edge endpoints.
    y : ndarray, shape [N']
        New labels.
    vertex_to_idx : dict
        Mapping from vertex name to its index in the arrays.
    label_to_idx : dict
        Mapping from label name to its value in `y`.
    mask : ndarray, shape [N]
        Mask indicating which vertices of the original graph are now part of the subgraph.
        That is, for example `x[mask] == x'`.
    """
    select_labels = set(select_labels)
    mask = get_label_mask(y, select_labels)

    x, edge_index, y, vertex_to_idx = graph_select_idxs(mask, 
            x, edge_index, y, vertex_to_idx)
    if connected:
        x, edge_index, y, vertex_to_idx, lcc_mask = graph_get_largest_connected_component(x, edge_index, y, vertex_to_idx)
        mask[mask] &= lcc_mask # Weird, isn't it?
    if _compress_labels:
        y, label_to_idx, _ = compress_labels(y, label_to_idx)
    return x, edge_index, y, vertex_to_idx, label_to_idx, mask
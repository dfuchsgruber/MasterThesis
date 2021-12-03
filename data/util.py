import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Dataset, Data
from data.base import SingleGraphDataset
import torch
from warnings import warn
import data.constants as data_constants

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
    # Create a hash-like object of all vertex features
    n, d = data.x.size()
    vertex_hash = data.x * torch.cos(torch.arange(0, n).repeat(d).view((d, n)).T / 1.5 **(2 * torch.arange(0, d).repeat(n).view((n, d)) / d))
    summary.append(f'{prefix}Vertex hash in mask: {vertex_hash[data.mask].mean()}')
    summary.append(f'{prefix}Vertex hash graph: {vertex_hash.mean()}')
    

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

def permute_label_to_idx_to_match(label_to_idx_base, label_to_idx_target):
    """ Reorders a source label mapping to respect the mapping defined in target. 
    That is, `label_to_idx_base` and `label_to_idx_target` are mappings name -> idx.
    This function returns a mapping that maps all names in `label_to_idx_base` to the same values
    as `label_to_idx_target`. The other names are mapped such that the mapping is compressed.
    
    Parameters:
    -----------
    label_to_idx_base : dict
        The mapping to reorder.
    label_to_idx_target : dict
        The new fixed mappings to apply.

    Returns:
    --------
    remapping : dict
        A mapping from idx_old to idx_new that can be fed into `remap_labels`.
    """
    remapping = dict()
    for label, idx_new in label_to_idx_target.items():
        if label in label_to_idx_base:
            remapping[label_to_idx_base[label]] = idx_new
    # Fill labels with a mapping that isnt fixed by `label_to_idx_target`
    for label, idx_old in label_to_idx_base.items():
        if idx_old not in remapping:
            idx_new = min(set(range(max(remapping.values()) + 2)) - set(remapping.values())) # Get the smallest unused idx
            remapping[idx_old] = idx_new
    return remapping


def labels_to_idx(labels, dataset):
    """ Gets the idxs of a certain label set within a dataset. 
    
    Parameters:
    -----------
    labels : iterable of int or str or 'all'
        The labels to get
    dataset : SingleGraphDataset
        The dataset in which to get labels from.
    
    Returns:
    --------
    labels : list of int
        Integer labels.
    """
    if labels == 'all':
        return list(dataset.label_to_idx.values())
    else:
        if all(isinstance(label, str) for label in labels):
            labels = [dataset.label_to_idx[label] for label in labels]
        elif all(isinstance(label, int) for label in labels):
            raise RuntimeError(f'Using ints to adress labels is deprecated!')
        if all(isinstance(label, int) for label in labels):
            return list(labels)
        else:
            raise RuntimeError(f'Could not understand the datatype of labels {set(type(label) for label in labels)}')

def uniform_split_with_fixed_test_set_portion(data, num_splits, num_train=20, num_val=20, portion_test_fixed=0.2, train_labels='all', train_labels_remove_other=False, 
        val_labels='all', val_labels_remove_other=False, base_labels='all'):
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
    base_labels : iterable or 'all'
        If given, the entire graph is reduced to these base labels before.

    Returns:
    --------
    data_list : list
        Different splits. Each split is indexed by `data.constants` constants.
    data_test_fixed : torch_geometric.data.Dataset
        Fixed test dataset that is consistent among all splits.
    """
    # Get the base graph from the dataset
    if base_labels != 'all':
        base_labels = labels_to_idx(base_labels, data)
        # Reduce the data before-hand: Note that this shifts labels, so it recommended to refer to them by string names
        x, edge_index, y, vertex_to_idx, label_to_idx, mask = graph_select_labels(data.x.numpy(), 
            data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, base_labels, 
            connected=True, _compress_labels=True)
        data = SingleGraphDataset(x, edge_index, y, vertex_to_idx, label_to_idx, np.ones(y.shape[0]).astype(bool))[0]

    # Remap labels such that the train labels take {0, .... num_train_classes - 1}
    # This way, training can be enabled
    train_label_compression = {label : idx for idx, label in enumerate(labels_to_idx(train_labels, data))}
    for name, label in data.label_to_idx.items():
        if label not in train_label_compression:
            train_label_compression[label] = len(train_label_compression)
    y, label_to_idx = remap_labels(data.y.numpy(), data.label_to_idx, train_label_compression)
    assert not (y == -1).any()
    data.y = torch.tensor(y).long()
    data.label_to_idx = label_to_idx

    seeds = data_split_seeds(num_splits + 1) # The first seed is used to fix the shared testing data
    
    # First stratified split to get the untouched test data
    mask_fixed, mask_non_fixed = stratified_split(data.y.numpy(), seeds[0:1], [portion_test_fixed, 1 - portion_test_fixed])[:, 0, :]

    all_labels = set(labels_to_idx('all', data))

    train_labels = set(labels_to_idx(train_labels, data))
    # Reduce the graph for training
    if train_labels_remove_other:
        x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train_graph = graph_select_labels(data.x.numpy(), 
            data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, train_labels, connected=True, _compress_labels=False)
    else:
        x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train_graph = (data.x.numpy(), 
            data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx.copy(), data.label_to_idx.copy(), torch.ones_like(data.y).bool())

    # Reduce graph for validation
    val_labels = set(labels_to_idx(val_labels, data))
    if val_labels_remove_other:
        x_val, edge_index_val, y_val, vertex_to_idx_val, label_to_idx_val, mask_val_graph = graph_select_labels(data.x.numpy(), 
            data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, val_labels, connected=True, _compress_labels=False)
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

        ### ---------- TRAINING ----------- ###
        # Sample train idxs
        mask_train = sample_uniformly(data.y.numpy(), train_labels, num_train, mask_train_graph & mask_non_fixed, rng=rng)
        data_train = SingleGraphDataset(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train[mask_train_graph])

        ### ---------- VALIDATION ----------- ###
        # A) Sample val reduced idxs (=validation vertices on the training label set for monitoring), disjunct from train idxs
        mask_val_reduced = sample_uniformly(data.y.numpy(), train_labels, num_val, mask_train_graph & (~mask_train) & mask_non_fixed, rng=rng)
        data_val_reduced = SingleGraphDataset(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_val_reduced[mask_train_graph])
        
        # B) Sample val idxs on the validation graph, disjunct from train idxs
        # Re-use the vertices from A) if there for classes that are both in training and validation labels
        # Sample val idxs: First sample idxs from the val classes not in the training classes
        mask_val = mask_val_reduced | sample_uniformly(data.y.numpy(), val_labels - train_labels, num_val, mask_val_graph & (~mask_train) & mask_non_fixed, rng=rng)
        # Remove train labels from the reduced mask that are not in val
        for label in train_labels - val_labels:
            mask_val[data.y.numpy() == label] = False
        data_val = SingleGraphDataset(x_val, edge_index_val, y_val, vertex_to_idx_val, label_to_idx_val, mask_val[mask_val_graph])

        # C) Get val idxs that are on the validation graph and also appear in the training graph
        # That is, this graph is potentially different from the training graph, but only masks vertices that have training labels
        mask_val_train_labels = mask_val.copy()
        for label in val_labels - train_labels:
            mask_val_train_labels[data.y.numpy() == label] = False
        # Permute labels so we match training data
        data_val_train_labels = SingleGraphDataset(x_val, edge_index_val, y_val, vertex_to_idx_val, label_to_idx_val, 
                                                        mask_val_train_labels[mask_val_graph])

        ### ------------ TEST ---------------- ###
        # A) Test idx are the non-used idxs (which are not fixed)
        mask_test = (~(mask_train | mask_val_reduced | mask_val)) & mask_non_fixed
        data_test = SingleGraphDataset(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, mask_test)
        
        # B) Test idxs on the training graph with train labels
        # Test idx for the reduced graph is the subset of the test graph that is on train labels, minus all non-training classes
        mask_test_reduced = mask_test.copy()
        for label in all_labels - train_labels:
            mask_test_reduced[data.y.numpy() == label] = False
        data_test_reduced = SingleGraphDataset(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_test_reduced[mask_train_graph])

        # C) Test idxs on the testing graph (=all labels), but only with train labels
        data_test_train_labels = SingleGraphDataset(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, mask_test_reduced)

        # Sanity checks
        assert ((mask_train.astype(int) + mask_val_reduced.astype(int)) <= 1).all(), f'Training and reduced Validation data are not disjunct'
        assert ((mask_train.astype(int) + mask_val.astype(int)) <= 1).all(), f'Training and full Validation data are not disjunct'
        assert ((mask_train.astype(int) + mask_test_reduced.astype(int)) <= 1).all(), f'Training and reduced Testing data are not disjunct'
        assert ((mask_train.astype(int) + mask_test.astype(int)) <= 1).all(), f'Training and full Testing data are not disjunct'
        assert (((mask_train | mask_val | mask_val_reduced).astype(int) + mask_test.astype(int)) <= 1).all(), f'Training & Validation and Testing data are not disjunct'
        assert (((mask_train | mask_val | mask_val_reduced | mask_test).astype(int) + mask_fixed.astype(int)) <= 1).all(), f'Non-fixed and fixed data are not disjunct'

        for idx, dataset in enumerate((data_train, data_val, data_val_reduced, data_test)):
            assert (dataset[0].x.size()[0] == dataset[0].mask.size()[0])
            assert (dataset[0].y.size() == dataset[0].mask.size())
            assert (dataset[0].edge_index <= dataset[0].x.size()[0]).all()

        data_list.append({
            data_constants.TRAIN : data_train,
            data_constants.VAL : data_val,
            data_constants.VAL_REDUCED : data_val_reduced,
            data_constants.VAL_TRAIN_LABELS : data_val_train_labels,
            data_constants.TEST : data_test,
            data_constants.TEST_REDUCED : data_test_reduced,
            data_constants.TEST_TRAIN_LABELS : data_test_train_labels,
        })

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

def remap_labels(labels, label_to_idx, remapping, undefined_label=-1):
    """ Remaps labels of a graph. 
    
    Parameters:
    -----------
    labels : ndarray, shape [N]
        Labels to remap.
    label_to_idx : dict
        A dict mapping a label name to its index.
    remapping : dict
        A mapping from label_old -> label_new.
    undefined_label : int
        Labels that are not in the remapping.
    
    Returns:
    --------
    labels : ndarray, shape [N]
        Remapped labels.
    label_to_idx : dict
        Mapping from label name to new index.
    """
    labels_new = np.ones_like(labels) * undefined_label
    label_to_idx_new = {}
    idx_to_label = {idx : label for label, idx in label_to_idx.items()}
    for label_old, label_new in remapping.items():
        labels_new[labels == label_old] = label_new
        label_to_idx_new[idx_to_label[label_old]] = label_new
    # assert not (labels_new == -1).any()
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
    # label_to_idx = {label : idx for label, idx in label_to_idx.items() if idx in select_labels}
    return x, edge_index, y, vertex_to_idx, label_to_idx, mask

def label_binarize(labels, num_classes=None):
    """ Transforms labels into a binary indicator matrix. That is, if `labels[i] == k`, 
    then `binarized[i, j] = 0` for all j != k and `binarized[i, k] = 1`.
    
    Parameters:
    -----------
    labels : torch.Tensor, shape [N]
        Labels to binarize.
    num_classes : int or None
        Number of classes. If `Ǹone`, it is inferred as `labels.max() + 1`

    Returns:
    --------
    binarized : torch.Tensor, shape [N, num_classes]
        Binary indicator matrix for labels.
    """
    if num_classes is None:
        num_classes = (labels.max() + 1).item()
    binarized = torch.zeros((labels.size(0), num_classes), device=labels.device).long()
    for i, label in enumerate(labels):
        binarized[i, label] = 1
    return binarized

def vertex_intersection(first, second):
    """ Finds the vertex intersection of two datasets that represent different subgraphs of the same supergraph.
    
    Parameters:
    -----------
    first : torch_geometric.data.Data
        The first graph.
    second : torch_geometric.data.Data
        The second graph.

    Returns:
    --------
    idxs_first : ndarray, shape [num_intersecting]
        Idxs of the intersecting vertices in the first graph.
    idxs_second : ndarray, shape [num_intersecting]
        Idxs of the intersecting vertices in the second graph.
    """
    idxs_first, idxs_second = [], []
    for vertex, idx_first in first.vertex_to_idx.items():
        if vertex in second.vertex_to_idx:
            idxs_first.append(idx_first.item())
            idxs_second.append(second.vertex_to_idx[vertex].item())
    return np.array(idxs_first, dtype=int), np.array(idxs_second, dtype=int)

def labels_in_dataset(data, mask=True):
    """ Returns all labels in a given dataset.
    
    Parameters:
    -----------
    data : torch_geometric.data.Data
        The single graph data.
    mask : bool
        If True, only vertices within the data's mask will be considered.
    
    Returns:
    --------
    labels_in_dataset : set
        All the labels in that dataset, with their real name (str) instead of the idx
        to make it comparable among different label compressions.
    """
    data = data.cpu()
    idx_to_label = {idx.item() : label for label, idx in data.label_to_idx.items()}
    if mask:
        return set(idx_to_label[idx] for idx in torch.unique(data.y[data.mask]).tolist())
    else:
        return set(idx_to_label[idx] for idx in torch.unique(data.y).tolist())

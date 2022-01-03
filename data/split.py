from typing import List
import numpy as np
import torch
from data.base import SingleGraphDataset
from warnings import warn
import data.constants as dconstants
import data.util as dutils
from copy import deepcopy

from seed import data_split_seeds, data_split_seeds_iterator

def _graph_drop(x, edge_index, y, vertex_to_idx, mask, mask_drop):
    """ Drops vertices from a graph and selects the lcc. 
    
    Parameters:
    -----------
    x : ndarray, shape [N, D]
        Attributes.
    edge_index : ndarray, shape [2, M]
        Edges.
    y : ndarray, shape [N]
        Labels
    vertex_to_idx : dict
        Mapping from vertex to idxs.
    mask : ndarray, shape [N*]
        Supermask that will be modified as well. E.g. this can be the mask that associates vertices in x with the supergraph.
        Thus, mask.sum() must equal N
    mask_drop : ndarray, [N]
        Which vertices in X to drop.

    Returns:
    --------
    x : ndarray, shape [N']
        New attributes.
    edge_index : ndarray, shape [2, M']
        New edges.
    y : ndarray, shape [N']
        New labels.
    vertex_to_idx : dict
        New mapping from vertex to idxs.
    mask : ndarray, shape [N*]
        Updated supermask to keep track of which vertices were dropped. Again, mask.sum() equals N'
    """

    mask = mask.copy()
    mask_kept = ~(mask_drop) # shape: [mask_graph.sum()] (i.e. indexes vertices on the train graph *before* dropping)
    x, edge_index, y, vertex_to_idx = dutils.graph_select_idxs(~(mask_drop), x, edge_index, y, vertex_to_idx)
    assert mask_kept.sum() == x.shape[0]
    x, edge_index, y, vertex_to_idx, mask_kept_lcc = dutils.graph_get_largest_connected_component(x, edge_index, y, vertex_to_idx)
    # Remove the dropped vertices from `mask_graph`. This is a bit confusing, but we need to "backpropagte" the mask through all the graph reduction steps
    mask_kept[mask_kept] &= mask_kept_lcc
    assert mask_kept.sum() == x.shape[0]
    mask[mask] &= mask_kept
    assert mask.sum() == x.shape[0]
    return x, edge_index, y, vertex_to_idx, mask

def uniform_split_perturbations(data, setting, num_splits, mask_non_fixed, seed_generator, num_samples, base_labels, train_labels, budget, drop_train, max_attempts_per_split = 5):
    """ Creates data splits for the a perturbation-based experiment. """ 
    # Both ID and OOD graphs are based on train-labels only (no left out classes)
    x_train_base, edge_index_train_base, y_train_base, vertex_to_idx_train_base, label_to_idx_train_base, mask_train_graph_base = dutils.graph_select_labels(data.x.numpy(), 
        data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, train_labels, connected=True, _compress_labels=False)
    
    data_list = []
    attempts = 0
    while len(data_list) < num_splits:
        if attempts >= max_attempts_per_split:
            raise RuntimeError(f'Could not generate split {len(data_list)} within {max_attempts_per_split} attempts.')
        rng = np.random.RandomState(next(seed_generator))
        try:
            # Build graphs used for training and ood-experiments
            x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train_graph = x_train_base.copy(), edge_index_train_base.copy(), y_train_base.copy(), deepcopy(vertex_to_idx_train_base), deepcopy(label_to_idx_train_base), mask_train_graph_base.copy()

            # Integrity assertions
            dutils.assert_integrity(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx,
                                x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, 
                                assert_second_is_vertex_subset=True, assert_second_is_label_subset=True)
            
            # Select vertices that will be perturbed and vertices that will be droppend to form an ood dataset
            # In a transductive setting, neither of those will be removed from the train graph
            _masks = dutils.split_from_mask_stratified(mask_train_graph, data.y.numpy(), sizes = [drop_train, budget, 1 - drop_train - budget], rng=rng)[:]
            mask_dropped, mask_perturbed = _masks[:, 0], _masks[:, 1]

            if setting in dconstants.TRANSDUCTIVE:
                mask_train = dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples, mask_train_graph & mask_non_fixed & (~mask_perturbed), rng=rng)
                data_train = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train[mask_train_graph])

                mask_val = dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples, mask_train_graph & (~mask_train) & mask_non_fixed, rng=rng)
                data_val = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_val[mask_train_graph])

                # Test are all non train and val vertices on the train graph with train labels
                mask_test = mask_train_graph & (~mask_train) & (~mask_val) & dutils.get_label_mask(data.y.numpy(), train_labels) 
                data_test = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_test[mask_train_graph])

                # The ood mask includes `num_samples` vertices per class that are not perturbed and `num_samples` perturbed vertices
                # Per class, `num_samples` are drawn, where 50% are perturbed and 50% are unperturbed
                mask_ood = dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples // 2, mask_train_graph & mask_perturbed & (~mask_train) & mask_non_fixed, rng=rng)
                mask_ood |= dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples - (num_samples // 2), mask_train_graph & (~mask_perturbed) & (~mask_train) & mask_non_fixed, rng=rng)
                data_ood = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_ood[mask_train_graph],
                    is_train_graph_vertex = np.ones(y_train.shape[0], dtype=bool), is_out_of_distribution = mask_perturbed[mask_train_graph])

                mask_ood_test = mask_train_graph & (~mask_train) & (~mask_ood) & (~mask_val) & dutils.get_label_mask(data.y.numpy(), train_labels)
                data_ood_test = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_ood_test[mask_train_graph], 
                    is_train_graph_vertex = np.ones(y_train.shape[0], dtype=bool), is_out_of_distribution = mask_perturbed[mask_train_graph])

            elif setting in dconstants.HYBRID:
                
                x_train, edge_index_train, y_train, vertex_to_idx_train, mask_train_graph = _graph_drop(
                    x_train, edge_index_train, y_train, vertex_to_idx_train, mask_train_graph, (mask_dropped | mask_perturbed)[mask_train_graph],    
                )
                x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, mask_ood_graph = x_train_base.copy(), edge_index_train_base.copy(), y_train_base.copy(), deepcopy(vertex_to_idx_train_base), mask_train_graph_base.copy()
                
                # Sample training, validation and testing : All of these have the same graph that is used for training.
                mask_train = dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples, mask_train_graph & mask_non_fixed, rng=rng)
                data_train = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train[mask_train_graph])

                mask_val = dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples, mask_train_graph & (~mask_train) & mask_non_fixed, rng=rng)
                data_val = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_val[mask_train_graph])

                # Test are all non train and val vertices on the train graph with train labels
                mask_test = mask_train_graph & (~mask_train) & (~mask_val) & dutils.get_label_mask(data.y.numpy(), train_labels) 
                data_test = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_test[mask_train_graph])

                # The ood mask includes `num_samples` vertices per class that are not perturbed and `num_samples` perturbed vertices
                # Per class, `num_samples` are drawn, where 50% are perturbed and 25% are unperturbed and not on the train graph and 25% are unperturbed on the train graph
                mask_ood = dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples // 2, mask_ood_graph & mask_perturbed & (~mask_train) & mask_non_fixed, rng=rng)
                mask_ood |= dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples // 4, mask_ood_graph & mask_train_graph & (~mask_perturbed) & (~mask_train) & mask_non_fixed, rng=rng)
                mask_ood |= dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples - (num_samples // 2) - (num_samples // 4), mask_ood_graph & (~mask_train_graph) & (~mask_perturbed) & (~mask_train) & mask_non_fixed, rng=rng)
                data_ood = SingleGraphDataset.build(x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_train, mask_ood[mask_ood_graph],
                    is_train_graph_vertex = mask_train_graph[mask_ood_graph], is_out_of_distribution = mask_perturbed[mask_ood_graph])

                mask_ood_test = mask_ood_graph & (~mask_train) & (~mask_ood) & (~mask_val) & dutils.get_label_mask(data.y.numpy(), train_labels)
                data_ood_test = SingleGraphDataset.build(x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_train, mask_ood_test[mask_ood_graph], 
                    is_train_graph_vertex = mask_train_graph[mask_ood_graph], is_out_of_distribution = mask_perturbed[mask_ood_graph])

            datasets = {
                dconstants.TRAIN : data_train,
                dconstants.VAL : data_val,
                dconstants.TEST : data_test,
                dconstants.OOD : data_ood,
                dconstants.OOD_TEST : data_ood_test,
            }

            # Furhter integrity assertcions
            assert ((mask_train.astype(int) + mask_val.astype(int)) <= 1).all(), f'Training and validation data are not disjunct'
            assert ((mask_train.astype(int) + mask_test.astype(int)) <= 1).all(), f'Training and test data are not disjunct'
            assert ((mask_val.astype(int) + mask_test.astype(int)) <= 1).all(), f'Validation and test data are not disjunct'

            assert ((mask_train.astype(int) + mask_ood.astype(int)) <= 1).all(), f'Training and ood data are not disjunct'
            assert ((mask_train.astype(int) + mask_ood_test.astype(int)) <= 1).all(), f'Training and ood test data are not disjunct'
            assert ((mask_ood.astype(int) + mask_ood_test.astype(int)) <= 1).all(), f'Ood data and ood test data are not disjunct'

            assert (((mask_train | mask_val).astype(int) + (~mask_non_fixed).astype(int)) <= 1).all(), f'Non-fixed and fixed data are not disjunct'
            assert (((mask_ood).astype(int) + (~mask_non_fixed).astype(int)) <= 1).all(), f'Ood data and fixed data are not disjunct'

            for name, dataset in datasets.items():
                assert (dataset[0].x.size()[0] == dataset[0].mask.size()[0]), name
                assert (dataset[0].y.size() == dataset[0].mask.size()), name
                assert (dataset[0].edge_index <= dataset[0].x.size()[0]).all(), name

            data_list.append(datasets)
            attempts = 0
            
        except dutils.SamplingError as e:
            attempts += 1
            warn(f'Split {len(data_list)} failed due to an sampling error in splitting: {e}. Trying next seed...')
    
    return data_list


def uniform_split_left_out_classes(data, setting, num_splits, mask_non_fixed, seed_generator, num_samples, train_labels, left_out_class_labels, drop_train, max_attempts_per_split = 5):
    """ Creates data splits for a left-out-classes experiment. """
    assert len(train_labels.intersection(left_out_class_labels)) == 0, f'Training labels and left out class labels can not intersect'
    ood_graph_labels = train_labels | left_out_class_labels
    if setting in dconstants.TRANSDUCTIVE:
        train_graph_labels = train_labels | left_out_class_labels
    elif setting in dconstants.HYBRID:
        train_graph_labels = train_labels
    else:
        raise RuntimeError(f'Unsupported setting for LoC experiment: {setting}')
    x_train_base, edge_index_train_base, y_train_base, vertex_to_idx_train_base, label_to_idx_train_base, mask_train_graph_base = dutils.graph_select_labels(data.x.numpy(), 
        data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, train_graph_labels, connected=True, _compress_labels=False)
    x_ood_base, edge_index_ood_base, y_ood_base, vertex_to_idx_ood_base, label_to_idx_ood_base, mask_ood_graph_base = dutils.graph_select_labels(data.x.numpy(), 
        data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, ood_graph_labels, connected=True, _compress_labels=False)

    data_list = []
    attempts = 0
    while len(data_list) < num_splits:
        if attempts >= max_attempts_per_split:
            raise RuntimeError(f'Could not generate split {len(data_list)} within {max_attempts_per_split} attempts.')
        rng = np.random.RandomState(next(seed_generator))
        try:
            # Build graphs used for training and ood-experiments
            x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train_graph = x_train_base.copy(), edge_index_train_base.copy(), y_train_base.copy(), deepcopy(vertex_to_idx_train_base), deepcopy(label_to_idx_ood_base), mask_train_graph_base.copy()
            x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_ood, mask_ood_graph = x_ood_base.copy(), edge_index_ood_base.copy(), y_ood_base.copy(), deepcopy(vertex_to_idx_ood_base), deepcopy(label_to_idx_ood_base), mask_ood_graph_base.copy()

            # Drop a fraction of all train vertices that will be eligible to the ood dataset.
            # In an hybrid setting, these vertices have to be dropped from the graph as well
            if setting in dconstants.HYBRID:
                mask_drop = dutils.split_from_mask_stratified(mask_train_graph, data.y.numpy(), sizes = [drop_train, 1 - drop_train], rng=rng)[:, 0]
                x_train, edge_index_train, y_train, vertex_to_idx_train, mask_train_graph = _graph_drop(
                    x_train, edge_index_train, y_train, vertex_to_idx_train, mask_train_graph, mask_drop[mask_train_graph],    
                )

            # Integrity assertions
            dutils.assert_integrity(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx,
                                x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, 
                                assert_second_is_vertex_subset=True, assert_second_is_label_subset=True)
            dutils.assert_integrity(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx,
                                x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_ood, 
                                assert_second_is_vertex_subset=True, assert_second_is_label_subset=True)

            # Sample training, validation and testing : All of these have the same graph that is used for training. The mask applies only to vertices with a label in `train_labels`
            mask_train = dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples, mask_train_graph & mask_non_fixed, rng=rng)
            data_train = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train[mask_train_graph])

            mask_val = dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples, mask_train_graph & (~mask_train) & mask_non_fixed, rng=rng)
            data_val = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_val[mask_train_graph])

            # Test are all non train and val vertices on the train graph with train labels
            mask_test = mask_train_graph & (~mask_train) & (~mask_val) & dutils.get_label_mask(data.y.numpy(), train_labels) 
            data_test = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_test[mask_train_graph])

            # Sample vertices for the ood dataset while ensuring that id classes come from both the training graph and outside the training graph in an hybrid setting
            if setting in dconstants.TRANSDUCTIVE:
                # Sample `num_samples` per id and ood class. All vertices were on the training graph
                mask_ood = dutils.sample_uniformly(data.y.numpy(), train_labels | left_out_class_labels, num_samples, mask_ood_graph & (~mask_train) & mask_non_fixed, rng=rng)
            elif setting in dconstants.HYBRID:
                # Sample `num_samples` per ood class, which are always not part of the training graph
                mask_ood = dutils.sample_uniformly(data.y.numpy(), left_out_class_labels, num_samples, mask_ood_graph & (~mask_train) & mask_non_fixed, rng=rng)
                # Sample `num_samples` per id class, where 50% come from the training graph and 50% come from outside the training graph
                mask_ood |= dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples // 2, mask_train_graph & mask_ood_graph & (~mask_train) & mask_non_fixed, rng=rng)
                mask_ood |= dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples - (num_samples // 2), (~mask_train_graph) & mask_ood_graph & (~mask_train) & mask_non_fixed, rng=rng)

            # Create the ood datasets: These include vertices from left out classes both in the graph and mask
            is_out_of_distribution = dutils.get_label_mask(data.y.numpy(), left_out_class_labels)

            # For the test portion of the ood graph, it can be assumed that it will include id vertices both on and not on the training graph in an hybrid setting
            mask_ood = dutils.sample_uniformly(data.y.numpy(), train_labels | left_out_class_labels, num_samples, mask_ood_graph & (~mask_train) & mask_non_fixed, rng=rng)
            data_ood = SingleGraphDataset.build(x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_ood, mask_ood[mask_ood_graph], 
                is_train_graph_vertex = mask_train_graph[mask_ood_graph], is_out_of_distribution = is_out_of_distribution[mask_ood_graph])

            mask_ood_test = mask_ood_graph & (~mask_train)& (~mask_val) & (~mask_ood) & dutils.get_label_mask(data.y.numpy(), train_labels | ood_graph_labels)
            data_ood_test = SingleGraphDataset.build(x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_ood, mask_ood_test[mask_ood_graph], 
                is_train_graph_vertex = mask_train_graph[mask_ood_graph], is_out_of_distribution = is_out_of_distribution[mask_ood_graph]) 

            datasets = {
                dconstants.TRAIN : data_train,
                dconstants.VAL : data_val,
                dconstants.TEST : data_test,
                dconstants.OOD : data_ood,
                dconstants.OOD_TEST : data_ood_test,
            }

            # Furhter integrity assertcions
            assert ((mask_train.astype(int) + mask_val.astype(int)) <= 1).all(), f'Training and validation data are not disjunct'
            assert ((mask_train.astype(int) + mask_test.astype(int)) <= 1).all(), f'Training and test data are not disjunct'
            assert ((mask_val.astype(int) + mask_test.astype(int)) <= 1).all(), f'Validation and test data are not disjunct'

            assert ((mask_train.astype(int) + mask_ood.astype(int)) <= 1).all(), f'Training and ood data are not disjunct'
            assert ((mask_train.astype(int) + mask_ood_test.astype(int)) <= 1).all(), f'Training and ood test data are not disjunct'
            assert ((mask_ood.astype(int) + mask_ood_test.astype(int)) <= 1).all(), f'Ood data and ood test data are not disjunct'

            assert (((mask_train | mask_val).astype(int) + (~mask_non_fixed).astype(int)) <= 1).all(), f'Non-fixed and fixed data are not disjunct'
            assert (((mask_ood).astype(int) + (~mask_non_fixed).astype(int)) <= 1).all(), f'Ood data and fixed data are not disjunct'

            for dataset in datasets.values():
                assert (dataset[0].x.size()[0] == dataset[0].mask.size()[0])
                assert (dataset[0].y.size() == dataset[0].mask.size())
                assert (dataset[0].edge_index <= dataset[0].x.size()[0]).all()

            data_list.append(datasets)
            attempts = 0
            
        except dutils.SamplingError as e:
            attempts += 1
            warn(f'Split {len(data_list)} failed due to an sampling error in splitting: {e}. Trying next seed...')

    return data_list


def uniform_split_with_fixed_test_portion(data, num_splits, num_samples=20, portion_test_fixed=0.2, train_labels='all', setting=dconstants.TRANSDUCTIVE[0], 
        left_out_class_labels='all', base_labels='all', drop_train = 0.0, perturbation_budget = 0.1, ood_type = dconstants.LEFT_OUT_CLASSES[0]):
    """ Splits the data in a uniform manner, i.e. it samples a fixed amount of vertices for traininig and validation. Remaining vertices are allocated to test.
    
    Parameters:
    -----------
    data : torch_geometric.data.Data
        The graph to split
    num_splits : int
        How many splits to generate
    portion_test_fixed : float
        A fraction of vertices that is fixed and will never be allocated to non-test datasets
    train_labels : 'all' or set
        The labels to train on
    setting : str
        The setting. Either transductive or hybrid.
    left_out_class_labels : 'all' or set
        In the LoC experiment, the labels of classes that are left out from training and treated as ood.
    base_labels : 'all' or set
        Before splitting, a connected graph will be cut from `data` selecting only those classes.
    drop_train : float
        A fraction of id training vertices that will be dropped from training, validation and testing graphs and be included in the ood graphs.
    perturbation_budget : float
        In the perturbation experiment, how many vertices will be marked for perturbations.
    ood_type : str
        Which experiment to conduct: Either left-out-classes or perturbations.

    Returns:
    --------
    data_list : list
        Each element is a dict representing an individual split.
    fixed_vertices : set
        The vertex ids (as in `data.vertex_to_idx`'s keys) of the common portion that will only be allocated to testing data. 
    """

    seed_generator = data_split_seeds_iterator()

    # Reduce the data before-hand: Note that this shifts labels, so it recommended to refer to them by string names
    x, edge_index, y, vertex_to_idx, label_to_idx, mask = dutils.graph_select_labels(data.x.numpy(), 
        data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, set(dutils.labels_to_idx(base_labels, data)), 
        connected=True, _compress_labels=True)
    data = SingleGraphDataset.build(x, edge_index, y, vertex_to_idx, label_to_idx, np.ones(y.shape[0]).astype(bool))[0]

    # Permute the labels of the base data such that the train labels are [0, ..., num_labels - 1]
    train_label_compression = {label : idx for idx, label in enumerate(set(dutils.labels_to_idx(train_labels, data)))}
    for name, label in data.label_to_idx.items():
        if label not in train_label_compression:
            train_label_compression[label] = len(train_label_compression)
    
    y, label_to_idx = dutils.remap_labels(data.y.numpy(), data.label_to_idx, train_label_compression)
    assert not (y == -1).any()
    data.y = torch.tensor(y).long()
    data.label_to_idx = label_to_idx

    # Calculate the numerical label sets only after the graph has been calculated and the labels reordered
    base_labels = set(dutils.labels_to_idx(base_labels, data))
    train_labels = set(dutils.labels_to_idx(train_labels, data))
    left_out_class_labels = set(dutils.labels_to_idx(left_out_class_labels, data))
    
    # Split away testing data
    mask_fixed, mask_non_fixed = dutils.stratified_split(data.y.numpy(), np.array([next(seed_generator)]), [portion_test_fixed, 1 - portion_test_fixed])[:, 0, :]
    if ood_type in dconstants.LEFT_OUT_CLASSES:
        return uniform_split_left_out_classes(data, setting, num_splits, mask_non_fixed, seed_generator, num_samples, train_labels, left_out_class_labels, drop_train), dutils.vertices_from_mask(mask_fixed, data.vertex_to_idx)
    elif ood_type in dconstants.PERTURBATION:
        return uniform_split_perturbations(data, setting, num_splits, mask_non_fixed, seed_generator, num_samples, base_labels, train_labels, perturbation_budget, drop_train), dutils.vertices_from_mask(mask_fixed & dutils.get_label_mask(data.y.numpy(), train_labels), data.vertex_to_idx)
    else:
        raise RuntimeError(f'Unsupported out-of-distribution type {ood_type}')
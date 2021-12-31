from typing import List
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Dataset, Data
from data.base import SingleGraphDataset
import torch
from warnings import warn
import data.constants as dconstants
import data.util as dutils
from copy import deepcopy

from seed import data_split_seeds, data_split_seeds_iterator


def uniform_split_perturbations(data, setting, num_splits, mask_non_fixed, seed_generator, num_samples, base_labels, train_labels, budget, drop_train, max_attempts_per_split = 5):
    pass

def uniform_split_left_out_classes(data, setting, num_splits, mask_non_fixed, seed_generator, num_samples, base_labels, train_labels, left_out_class_labels, drop_train, max_attempts_per_split = 5):
    """ Creates data splits for a left-out-classes experiment. """
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
            mask_drop_train = dutils.split_from_mask_stratified(mask_train_graph, data.y.numpy(), sizes = [drop_train, 1 - drop_train], rng=rng)[:, 0]
            if setting in dconstants.HYBRID:
                mask_kept = ~(mask_drop_train[mask_train_graph]) # shape: [mask_train_graph.sum()] (i.e. indexes vertices on the train graph *before* dropping)
                x_train, edge_index_train, y_train, vertex_to_idx_train = dutils.graph_select_idxs(~(mask_drop_train[mask_train_graph]), x_train, edge_index_train, y_train, vertex_to_idx_train)
                assert mask_kept.sum() == x_train.shape[0]
                x_train, edge_index_train, y_train, vertex_to_idx_train, mask_kept_lcc = dutils.graph_get_largest_connected_component(x_train, edge_index_train, y_train, vertex_to_idx_train)
                # Remove the dropped vertices from `mask_train_graph`. This is a bit confusing, but we need to "backpropagte" the mask through all the graph reduction steps
                mask_kept[mask_kept] &= mask_kept_lcc
                assert mask_kept.sum() == x_train.shape[0]
                mask_train_graph[mask_train_graph] &= mask_kept
                assert mask_train_graph.sum() == x_train.shape[0]
            
            # Integrity assertions
            dutils.assert_integrity(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx,
                                x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, 
                                assert_second_is_vertex_subset=True, assert_second_is_label_subset=True)
            dutils.assert_integrity(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx,
                                x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_ood, 
                                assert_second_is_vertex_subset=True, assert_second_is_label_subset=True)

            # Sample training, validation and testing : All of these have the same graph that is used for training. The mask applies only to vertices with a label in `train_labels`
            mask_train = dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples, mask_train_graph & mask_non_fixed, rng=rng)
            data_train = SingleGraphDataset(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_train[mask_train_graph])

            mask_val = dutils.sample_uniformly(data.y.numpy(), train_labels, num_samples, mask_train_graph & (~mask_train) & mask_non_fixed, rng=rng)
            data_val = SingleGraphDataset(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_val[mask_train_graph])

            # Test are all non train and val vertices on the train graph with train labels
            mask_test = mask_train_graph & (~mask_train) & (~mask_val) & mask_non_fixed & dutils.get_label_mask(data.y.numpy(), train_labels) 
            data_test = SingleGraphDataset(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_train, mask_test[mask_train_graph])

            # Create the ood datasets: These include vertices from left out classes both in the graph and mask
            mask_ood = dutils.sample_uniformly(data.y.numpy(), train_labels | left_out_class_labels, num_samples, mask_ood_graph & (~mask_train) & mask_non_fixed, rng=rng)
            data_ood = SingleGraphDataset(x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_ood, mask_ood[mask_ood_graph])

            mask_ood_test = mask_ood_graph & (~mask_train) & (~mask_ood) & mask_non_fixed & dutils.get_label_mask(data.y.numpy(), train_labels | ood_graph_labels)
            data_ood_test = SingleGraphDataset(x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_ood, mask_ood_test[mask_ood_graph]) 

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

            assert (((mask_train | mask_val | mask_test).astype(int) + (~mask_non_fixed).astype(int)) <= 1).all(), f'Non-fixed and fixed data are not disjunct'
            assert (((mask_ood | mask_ood_test).astype(int) + (~mask_non_fixed).astype(int)) <= 1).all(), f'Ood data and fixed data are not disjunct'

            for dataset in datasets.values():
                assert (dataset[0].x.size()[0] == dataset[0].mask.size()[0])
                assert (dataset[0].y.size() == dataset[0].mask.size())
                assert (dataset[0].edge_index <= dataset[0].x.size()[0]).all()

            data_list.append(datasets)
            attempts = 0
            
        except dutils.SamplingError as e:
            attempts += 1
            warn(f'Split {len(data_list)} failed due to an sampling error in splitting: {e}. Trying next seed...')
    
    data_test_fixed = SingleGraphDataset(x_train_base, edge_index_train_base, y_train_base, vertex_to_idx_train_base, label_to_idx_train_base, (~mask_non_fixed)[mask_train_graph_base] & dutils.get_label_mask(data.y.numpy(), train_labels) )
    data_ood_test_fixed = SingleGraphDataset(x_ood_base, edge_index_ood_base, y_ood_base, vertex_to_idx_ood_base, label_to_idx_ood_base, (~mask_non_fixed)[mask_ood_graph_base])
    
    return data_list, data_test_fixed, data_ood_test_fixed


def uniform_split_with_fixed_test_portion(data, num_splits, num_samples=20, portion_test_fixed=0.2, train_labels='all', setting=dconstants.TRANSDUCTIVE[0], 
        left_out_class_labels='all', base_labels='all', drop_train = 0.0, perturbation_budget = 0.1, ood_type = dconstants.LEFT_OUT_CLASSES[0]):

    seeder = data_split_seeds_iterator()

    # Reduce the data before-hand: Note that this shifts labels, so it recommended to refer to them by string names
    x, edge_index, y, vertex_to_idx, label_to_idx, mask = dutils.graph_select_labels(data.x.numpy(), 
        data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, set(dutils.labels_to_idx(base_labels, data)), 
        connected=True, _compress_labels=True)
    data = SingleGraphDataset(x, edge_index, y, vertex_to_idx, label_to_idx, np.ones(y.shape[0]).astype(bool))[0]

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
    mask_fixed, mask_non_fixed = dutils.stratified_split(data.y.numpy(), np.array([next(seeder)]), [portion_test_fixed, 1 - portion_test_fixed])[:, 0, :]
    if ood_type in dconstants.LEFT_OUT_CLASSES:
        return uniform_split_left_out_classes(data, setting, num_splits, mask_non_fixed, seeder, num_samples, base_labels, train_labels, left_out_class_labels, drop_train)
    else:
        raise RuntimeError(f'Unsupported out-of-distribution type {ood_type}')
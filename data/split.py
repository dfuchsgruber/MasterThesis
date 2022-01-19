import numpy as np
import torch
import torch_geometric as tg
from data.base import SingleGraphDataset
from warnings import warn
import data.constants as dconstants
import data.util as dutils
from copy import deepcopy
import configuration

from seed import data_split_seeds_iterator

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


def uniform_split_with_fixed_test_portion(data: tg.data.Data, num_splits: int, config: configuration.DataConfiguration):
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
    ood_sampling_strategy : str
        If the LoC split, how to sample vertices for the `dconstants.OOD[0]` dataset.
    max_num_attempts_per_split : int
        How many times a different seed can be redrawn for a given split.

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
        data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, set(dutils.labels_to_idx(config.base_labels, data)), 
        connected=True, _compress_labels=True)
    data = SingleGraphDataset.build(x, edge_index, y, vertex_to_idx, label_to_idx, np.ones(y.shape[0]).astype(bool))[0]
    mask_fixed, mask_non_fixed = dutils.stratified_split(data.y.numpy(), np.array([next(seed_generator)]), [config.test_portion_fixed, 1 - config.test_portion_fixed])[:, 0, :]

    # Permute the labels of the base data such that the train labels are [0, ..., num_labels - 1]
    train_label_compression = {label : idx for idx, label in enumerate(set(dutils.labels_to_idx(config.train_labels, data)))}
    for name, label in data.label_to_idx.items():
        if label not in train_label_compression:
            train_label_compression[label] = len(train_label_compression)
    
    y, label_to_idx = dutils.remap_labels(data.y.numpy(), data.label_to_idx, train_label_compression)
    assert not (y == -1).any(), f'Something wrent wrong in the relabeling of the data...'
    data.y = torch.tensor(y).long()
    data.label_to_idx = label_to_idx

    # Calculate the numerical label sets only after the graph has been calculated and the labels reordered
    base_labels = set(dutils.labels_to_idx(config.base_labels, data))
    train_labels = set(dutils.labels_to_idx(config.train_labels, data))
    left_out_class_labels = set(dutils.labels_to_idx(config.left_out_class_labels, data))
    assert base_labels == (train_labels | left_out_class_labels), f'Base labels is not the union of train labels and left out class labels'
    assert not train_labels.intersection(left_out_class_labels), f'Train labels and left out class labels intersect'
    if config.ood_type == dconstants.LEFT_OUT_CLASSES:
        assert len(left_out_class_labels) > 0, f'For a LoC experiment, left out classes should be specified'
    elif config.ood_type == dconstants.PERTURBATION:
        assert not left_out_class_labels, f'For a Perturbations experiment, no left out classes should be specified'

    x_base, edge_index_base, y_base, vertex_to_idx_base, label_to_idx_base, _ = dutils.graph_select_labels(data.x.numpy(), 
        data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, base_labels, connected=True, _compress_labels=False)

    splits = []
    while len(splits) < num_splits:
        for _ in range(config.max_attempts_per_split):
            try:
                rng = np.random.RandomState(next(seed_generator))

                # Decide which vertices will be labeled ood
                if config.ood_type == dconstants.LEFT_OUT_CLASSES:
                    is_ood_base = dutils.get_label_mask(y_base, left_out_class_labels)
                elif config.ood_type == dconstants.PERTURBATION:
                    is_ood_base = dutils.split_from_mask_stratified(np.ones_like(y_base, dtype=bool), y_base, sizes = [config.perturbation_budget, 1 - config.perturbation_budget], rng=rng)[:, 0]

                # Create a graph for training, validation, testing and one for the ood experiments (validation and testing)
                mask_dropped_id = dutils.split_from_mask_stratified(~is_ood_base, y_base, sizes = [config.drop_train_vertices_portion, 1 - config.drop_train_vertices_portion], rng=rng)[:, 0]
                if config.setting == dconstants.TRANSDUCTIVE:
                    x_train, edge_index_train, y_train, vertex_to_idx_train, mask_train_graph = (x_base.copy(), edge_index_base.copy(), y_base.copy(), 
                    deepcopy(vertex_to_idx_base), np.ones_like(y_base, dtype=bool))
                elif config.setting == dconstants.HYBRID:
                    # Drop a portion of all in-distribution vertices from the train graph together with the out-of-distribution vertices
                    x_train, edge_index_train, y_train, vertex_to_idx_train, mask_train_graph = _graph_drop(
                        x_base, edge_index_base, y_base, vertex_to_idx_base, np.ones_like(y_base, dtype=bool), mask_dropped_id | is_ood_base,    
                    )
                    # After dropping, other in-distribution vertices may have been droppped during lcc selection,
                    # we include those as well into `mask_dropped_id`
                    additionally_dropped_id = (~mask_train_graph) & (~mask_dropped_id) & (~is_ood_base)
                    assert set(y_base[additionally_dropped_id]).issubset(train_labels)
                    # print(f'Additional {additionally_dropped_id.sum()} id vertices were dropped after lcc.')
                    mask_dropped_id |= additionally_dropped_id
                else:
                    raise ValueError

                x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, mask_ood_graph = (x_base.copy(), edge_index_base.copy(), y_base.copy(), 
                deepcopy(vertex_to_idx_base), np.ones_like(y_base, dtype=bool))

                # Sample dataset masks
                mask_train = dutils.sample_uniformly(y_base, train_labels, config.train_portion, mask_train_graph & ~is_ood_base & ~mask_dropped_id & mask_non_fixed, rng=rng)
                data_train = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_base, mask_train[mask_train_graph])

                mask_val = dutils.sample_uniformly(y_base, train_labels, config.val_portion, mask_train_graph & ~is_ood_base & ~mask_dropped_id & (~mask_train) & mask_non_fixed, rng=rng)
                data_val = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_base, mask_val[mask_train_graph])

                # TODO: Potentially, use a sampling strategy here as well?
                mask_test = mask_train_graph & ~is_ood_base & ~mask_dropped_id & mask_fixed
                data_test = SingleGraphDataset.build(x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_base, mask_test[mask_train_graph])

                # The ood masks are exclusively among vertices that were marked as either 
                #   i) ood or 
                #   ii) id and marked for dropping (they only actually *were* dropped in a hybrid setting)
                if config.ood_sampling_strategy == dconstants.SAMPLE_UNIFORM:
                    # In LoC, sample the ood-classes from the ood-mask and the id-classes from the dropped id-vertices
                    if config.ood_type == dconstants.LEFT_OUT_CLASSES:
                        mask_ood_val = dutils.sample_uniformly(y_base, left_out_class_labels, config.val_portion, mask_ood_graph & is_ood_base & mask_non_fixed, rng=rng)
                        mask_ood_val |= dutils.sample_uniformly(y_base, train_labels, config.val_portion, mask_ood_graph & mask_dropped_id & mask_non_fixed, rng=rng)
                    # In perturbations, sample 50% from the perturbed and %50 from the dropped non perturbed
                    elif config.ood_type == dconstants.PERTURBATION:
                        mask_ood_val = dutils.sample_uniformly(y_base, train_labels, config.val_portion // 2, mask_ood_graph & is_ood_base & mask_non_fixed, rng=rng)
                        mask_ood_val |= dutils.sample_uniformly(y_base, train_labels, config.val_portion - (config.val_portion // 2), mask_ood_graph & mask_dropped_id & mask_non_fixed, rng=rng)
                elif config.ood_sampling_strategy == dconstants.SAMPLE_ALL:
                    mask_ood_val = mask_ood_graph & (is_ood_base | mask_dropped_id) & mask_non_fixed
                else:
                    raise ValueError
                data_ood_val = SingleGraphDataset.build(x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_base, mask_ood_val[mask_ood_graph],
                    is_out_of_distribution=is_ood_base[mask_ood_graph], is_train_graph_vertex = mask_train_graph[mask_ood_graph])

                # TODO: Potentially, use the same sampling strategy as well?
                #  Note that the test-ood set may have not `num_samples` vertices per class in any split, so we might need to reduce the actual number
                #  of samples to the minimal class count
                mask_ood_test = (is_ood_base | mask_dropped_id) & mask_fixed
                data_ood_test = SingleGraphDataset.build(x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_base, mask_ood_test[mask_ood_graph],
                    is_out_of_distribution=is_ood_base[mask_ood_graph], is_train_graph_vertex = mask_train_graph[mask_ood_graph])

                datasets = {
                    dconstants.TRAIN : data_train,
                    dconstants.VAL : data_val,
                    dconstants.TEST : data_test,
                    dconstants.OOD_VAL : data_ood_val,
                    dconstants.OOD_TEST : data_ood_test,
                }

                # Checks and integrity assertions

                dutils.assert_integrity(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx,
                                    x_train, edge_index_train, y_train, vertex_to_idx_train, label_to_idx_base, 
                                    assert_second_is_vertex_subset=True, assert_second_is_label_subset=True)
                dutils.assert_integrity(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx,
                                    x_ood, edge_index_ood, y_ood, vertex_to_idx_ood, label_to_idx_base, 
                                    assert_second_is_vertex_subset=True, assert_second_is_label_subset=True)

                assert ((mask_train.astype(int) + mask_val.astype(int)) <= 1).all(), f'Training and validation data are not disjunct'
                assert ((mask_train.astype(int) + mask_test.astype(int)) <= 1).all(), f'Training and test data are not disjunct'
                assert ((mask_val.astype(int) + mask_test.astype(int)) <= 1).all(), f'Validation and test data are not disjunct'

                assert ((mask_train.astype(int) + mask_ood_val.astype(int)) <= 1).all(), f'Training and ood data are not disjunct'
                assert ((mask_train.astype(int) + mask_ood_test.astype(int)) <= 1).all(), f'Training and ood test data are not disjunct'
                assert ((mask_ood_val.astype(int) + mask_ood_test.astype(int)) <= 1).all(), f'Ood data and ood test data are not disjunct'

                assert (((mask_train | mask_val).astype(int) + (mask_fixed).astype(int)) <= 1).all(), f'Non-fixed and fixed data are not disjunct'
                assert (((mask_ood_val).astype(int) + (mask_fixed).astype(int)) <= 1).all(), f'Ood data and fixed data are not disjunct'

                for name, dataset in datasets.items():
                    assert (dataset[0].x.size()[0] == dataset[0].mask.size()[0]), name
                    assert (dataset[0].y.size() == dataset[0].mask.size()), name
                    assert (dataset[0].edge_index <= dataset[0].x.size()[0]).all(), name

                splits.append(datasets)
                break

            except dutils.SamplingError as e: 
                warn(f'Split {len(splits)} failed due to an sampling error in splitting: {e}. Trying next seed...')
        else:
            raise RuntimeError(f'Could not generate split {len(splits)} after {config.max_attempts_per_split} attempts!')

    return splits, dutils.vertices_from_mask(mask_fixed, data.vertex_to_idx)




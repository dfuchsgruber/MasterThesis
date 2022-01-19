import numpy as np
import torch

from data.gust_dataset import GustDataset
from data.npz import NpzDataset
from torch_geometric.transforms import BaseTransform, Compose
from data.transform import MaskTransform, MaskLabelsTransform, RemoveLabelsTransform
from data.util import stratified_split_with_fixed_test_set_portion
from data.split import uniform_split_with_fixed_test_portion
import data.constants
from configuration import DataConfiguration

def _append_labels_transform(dataset, select_labels, remove_other_labels, compress_labels):
    """ Appends a transformation that only selects certain labels. 
    
    Parameters:
    -----------
    dataset : torch_geometric.data.Dataset
        The dataset to append the transformation to.
    select_labels : iterable
        The labels to select. Other labels will be excluded.
    remove_other_labels : bool
        If `True`, vertices (and edges) with labels not in `select_labels` will be removed and the
        largest connected component of the remaining graph will be used.
        If `False`, vertices not in `select_labels` will be removed from the the vertex mask of each input.
    compress_labels : bool
        If `True`, labels will be remapped to (0, 1, ..., k) after masking / deletion.
    """
    if remove_other_labels:
        dataset.transform = Compose([
            dataset.transform, RemoveLabelsTransform(select_labels, compress_labels=compress_labels)
        ])
    else:
        dataset.transform = Compose([
            dataset.transform, MaskLabelsTransform(select_labels, compress_labels=compress_labels)
        ])

def _load_gust_data_from_configuration(config):
    """ Helper to load and split gust datasets. """
    raise NotImplemented

def _print_stats(dataset, prefix=''):
    d = dataset[0]
    print(f'{prefix}Number Vertices : {d.x[d.mask].size()[0]}')
    print(f'{prefix}Average postional feature : {(d.x[d.mask] * torch.arange(d.mask.sum()).view(-1, 1) ).mean()}')
    print(f'{prefix}Labels in underlying graph : {torch.unique(d.y)}')
    print(f'{prefix}Average postional feature in underlying graph : {(d.x * torch.arange(d.x.size()[0]).view(-1, 1) ).mean()}')
    print(f'{prefix}Number vertices in underlying graph : {d.x.size()[0]}')
    # Class distribution
    distr_str = 'Class distribution: '
    for label, cnt in zip(*np.unique(d.y[d.mask].numpy(), return_counts=True)):
        distr_str += f'{label} : {cnt} ({cnt/d.mask.sum().item() * 100 :.2f}%), '
    print(f'{prefix}{distr_str}')

def _print_datasets_stats(datasets, prefix=''):
    for idx, name in ((0, 'Training'), (1, 'Validation-Reduced'), (2, 'Validation Full'), (3, 'Testing')):
        print(f'{prefix} ## {name} ##')
        _print_stats(datasets[idx], prefix=f'{prefix}\t')

def load_data_from_configuration_uniform_split(config: DataConfiguration, num_splits):
    """ Loads datasets from a configuration and splits it according to the global split by uniformyl sampling for a fixed number of vertices per class.
    
    Parameters:
    -----------
    config : dict
        The configuration from which to load data from.
    num_splits : int
        How many splits to create.
    
    Returns:
    --------
    data_list : list
        A list of different data splits. Each element is 4-tuple of data_train, data_val, data_val_all_classes, data_test.
    data_test_fixed : torch_geometric.data.Dataset
        Fixed test dataset that is consistent among all splits.
    """
    if config.type in ('gust',):
        base_data = GustDataset(config.dataset)[0]
    elif config.type in ('npz',):
        base_data = NpzDataset.build(config)[0]
    else:
        raise RuntimeError(f'Unsupported dataset type {config.type}')

    return uniform_split_with_fixed_test_portion(
        base_data,
        num_splits,
        config
        )

def load_data_from_configuration(config: DataConfiguration, num_splits):
    """ Loads datasets from a configuration and splits it according to the global split.
    
    Parameters:
    -----------
    config : dict
        The configuration from which to load data from.
    num_splits : int
        How many splits to create.
    
    Returns:
    --------
    data_list : list
        A list of different data splits. Each element is 4-tuple of data_train, data_val, data_val_all_classes, data_test.
    fixed_vertices : set
        The vertex ids (as in `data.vertex_to_idx`'s keys) of the common portion that will only be allocated to testing data. 
    """
    if config.split_type == 'uniform':
        return load_data_from_configuration_uniform_split(config, num_splits)
    else:
        raise RuntimeError(f'Unsupported dataset split type {config.split_type}')
    
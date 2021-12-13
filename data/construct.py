import numpy as np
import torch

from data.gust_dataset import GustDataset
from data.npz import NpzDataset
from torch_geometric.transforms import BaseTransform, Compose
from data.transform import MaskTransform, MaskLabelsTransform, RemoveLabelsTransform
from data.util import stratified_split_with_fixed_test_set_portion, uniform_split_with_fixed_test_set_portion
import data.constants

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
    base_data = GustDataset(config['dataset'])[0]
    mask, mask_test_fixed = stratified_split_with_fixed_test_set_portion(base_data.y.numpy(),  config['num_dataset_splits'],
                                                           portion_train=config['train_portion'], 
                                                           portion_val=config['val_portion'], 
                                                           portion_test_fixed=config['test_portion_fixed'], 
                                                           portion_test_not_fixed=config['test_portion'],
                                                           )
    return [
            {
                name : GustDataset(config['dataset'], transform = MaskTransform(mask[type_idx, split_idx])) for name, type_idx in {
                    data.constants.TRAIN : 0,
                    data.constants.VAL : 1,
                    data.constants.VAL_REDUCED : 1,
                    data.constants.TEST : 2,
                    data.constants.TEST_REDUCED : 2,
                }.items()
            } for split_idx in range(mask.shape[1])
        ], GustDataset(config['dataset'], transform=MaskTransform(mask_test_fixed))

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

def load_data_from_configuration_stratified_split(config):
    """ Loads datasets from a configuration and splits in a stratified manner it according to the global split.
    
    Parameters:
    -----------
    config : dict
        The configuration from which to load data from.
    
    Returns:
    --------
    data_list : list
        A list of different data splits. Each element is 4-tuple of data_train, data_val, data_val_all_classes, data_test.
    data_test_fixed : torch_geometric.data.Dataset
        Fixed test dataset that is consistent among all splits.
    """
    if config['type'].lower() in ('gust',):
        data_list, dataset_fixed = _load_gust_data_from_configuration(config)
    else:
        raise RuntimeError(f'Unsupported dataset type {config["type"]}')

    # _print_datasets_stats(data_list[0])

    if config.get('train_labels', 'all') != 'all':
        # Select only certain labels from training data
        remove_other_labels = config.get('train_labels_remove_other', False)
        compress_labels = config.get('train_labels_compress', True)
        print(f'Reducing train labels to {config["train_labels"]}.\n\tRemove other vertices: {remove_other_labels}.\n\tCompressing labels: {compress_labels}.')
        for datasets in data_list:
            _append_labels_transform(datasets[data.constants.TRAIN], config['train_labels'], remove_other_labels, compress_labels)
            _append_labels_transform(datasets[data.constants.VAL_REDUCED], config['train_labels'], remove_other_labels, compress_labels)
            _append_labels_transform(datasets[data.constants.TEST_REDUCED], config['train_labels'], remove_other_labels, compress_labels)

    if config.get('val_labels', 'all') != 'all':
        # Select only certain labels from validation data
        remove_other_labels = config.get('val_labels_remove_other', False)
        compress_labels = config.get('val_labels_compress', True)
        print(f'Reducing val labels to {config["val_labels"]}.\n\tRemove other vertices: {remove_other_labels}.\n\tCompressing labels: {compress_labels}.')
        for datasets in data_list:
            _append_labels_transform(datasets[data.constants.VAL], config['val_labels'], remove_other_labels, compress_labels)

    return data_list, dataset_fixed

def load_data_from_configuration_uniform_split(config):
    """ Loads datasets from a configuration and splits it according to the global split by uniformyl sampling for a fixed number of vertices per class.
    
    Parameters:
    -----------
    config : dict
        The configuration from which to load data from.
    
    Returns:
    --------
    data_list : list
        A list of different data splits. Each element is 4-tuple of data_train, data_val, data_val_all_classes, data_test.
    data_test_fixed : torch_geometric.data.Dataset
        Fixed test dataset that is consistent among all splits.
    """
    if config['type'].lower() in ('gust',):
        base_data = GustDataset(config['dataset'])[0]
    elif config['type'].lower() in ('npz',):
        base_data = NpzDataset(
            config['dataset'], corpus_labels=config['corpus_labels'], min_token_frequency=config['min_token_frequency'],
            preprocessing = config['preprocessing'], language_model=config['language_model'],
        )[0]
    else:
        raise RuntimeError(f'Unsupported dataset type {config["type"]}')

    return uniform_split_with_fixed_test_set_portion(base_data,
        config['num_dataset_splits'],
        num_train = int(config['train_portion']),
        num_val = int(config['val_portion']),
        portion_test_fixed = config['test_portion_fixed'],
        train_labels = config.get('train_labels', 'all'),
        train_labels_remove_other=config.get('train_labels_remove_other', False),
        val_labels = config.get('val_labels', 'all'),
        val_labels_remove_other = config['val_labels_remove_other'],
        base_labels = config.get('base_labels', 'all'),
        )

def load_data_from_configuration(config):
    """ Loads datasets from a configuration and splits it according to the global split.
    
    Parameters:
    -----------
    config : dict
        The configuration from which to load data from.
    
    Returns:
    --------
    data_list : list
        A list of different data splits. Each element is 4-tuple of data_train, data_val, data_val_all_classes, data_test.
    data_test_fixed : torch_geometric.data.Dataset
        Fixed test dataset that is consistent among all splits.
    """
    split_type = config.get('split_type', 'stratified').lower()
    if split_type == 'stratified':
        data_list, data_test_fixed = load_data_from_configuration_stratified_split(config)
    elif split_type == 'uniform':
        data_list, data_test_fixed = load_data_from_configuration_uniform_split(config)
    else:
        raise RuntimeError(f'Unsupported dataset split type {split_type}')
    return data_list, data_test_fixed
    
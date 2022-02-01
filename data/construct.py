import numpy as np
import torch
from typing import Tuple, Dict

from data.gust_dataset import GustDataset
from data.npz import NpzDataset
from data.split import uniform_split_with_fixed_test_portion
from configuration import DataConfiguration
from torch_geometric.data import Dataset

def load_data_from_configuration_uniform_split(config: DataConfiguration, split_seed: int) -> Tuple[Dict[str, Dataset], set]:
    """ Loads datasets from a configuration and splits it according to the global split by uniformyl sampling for a fixed number of vertices per class.
    
    Parameters:
    -----------
    config : dict
        The configuration from which to load data from.
    split_seed : int
        The seed for splitting.
    
    Returns:
    --------
    data_dict : dict
        All datasets.
    fixed_vertices : set
        The vertex ids (as in `data.vertex_to_idx`'s keys) of the common portion that will only be allocated to testing data. 
    """
    if config.type in ('gust',):
        base_data = GustDataset(config.dataset)[0]
    elif config.type in ('npz',):
        base_data = NpzDataset.build(config)[0]
    else:
        raise RuntimeError(f'Unsupported dataset type {config.type}')

    return uniform_split_with_fixed_test_portion(
        base_data,
        split_seed,
        config
        )

def load_data_from_configuration(config: DataConfiguration, split_seed: int) -> Tuple[Dict[str, Dataset], set]:
    """ Loads datasets from a configuration and splits it according to the global split.
    
    Parameters:
    -----------
    config : dict
        The configuration from which to load data from.
    split_seed : int
        The seed for splitting.
    
    Returns:
    --------
    data_dict : dict
        All datasets.
    fixed_vertices : set
        The vertex ids (as in `data.vertex_to_idx`'s keys) of the common portion that will only be allocated to testing data. 
    """
    if config.split_type == 'uniform':
        return load_data_from_configuration_uniform_split(config, split_seed)
    else:
        raise RuntimeError(f'Unsupported dataset split type {config.split_type}')
    
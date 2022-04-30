from dataset_registry import DatasetRegistry
import numpy as np
import torch
from typing import Tuple, Dict
import logging

from data.gust_dataset import GustDataset
from data.npz import NpzDataset
from data.split import uniform_split_with_fixed_test_portion, predefined_split
from configuration import DataConfiguration
from torch_geometric.data import Dataset, Data

def load_base_data_from_configuration(config: DataConfiguration) -> Data:
    """ Loads the base data from the configuration. 
    
    Parameters:
    -----------
    config : DataConfiguration
        The configuration from which to load data from.
    
    Returns:
    --------
    data : Data
        The base data.
    """
    if config.type in ('gust',):
        base_data = GustDataset(config.dataset)[0]
    elif config.type in ('npz',):
        base_data = NpzDataset.build(config)[0]
    else:
        raise RuntimeError(f'Unsupported dataset type {config.type}')
    return base_data

def load_data_from_configuration_uniform_split(config: DataConfiguration, split_seed: int) -> Tuple[Dict[str, Dataset], set]:
    """ Loads datasets from a configuration and splits it according to the global split by uniformyl sampling for a fixed number of vertices per class.
    
    Parameters:
    -----------
    config : DataConfiguration
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

    return uniform_split_with_fixed_test_portion(
        load_base_data_from_configuration(config),
        split_seed,
        config
        )

def load_data_from_configuration_predefined_split(config: DataConfiguration, split_seed: int) -> Tuple[Dict[str, Dataset], set]:
    """ Loads a dataset with a predefined split. The split seed is used to determine which id-class vertices to drop in a hybrid setting. 
    
    Paramters:
    ----------
    config : DataConfiguration
        The configuration from which to load data from.
    split_seed : int
        Split seed to determine id vertices to drop in a hybrid setting.

    Returns:
    --------
    data_dict : dict
        All datasets.
    fixed_vertices : set
        The vertex ids (as in `data.vertex_to_idx`'s keys) of the common portion that will only be allocated to testing data. 
    """
    return predefined_split(
        load_base_data_from_configuration(config),
        split_seed,
        config
        )

def load_data_from_configuration(config: DataConfiguration, split_seed: int) -> Tuple[Dict[str, Dataset], set]:
    """ Loads datasets from a configuration and splits it according to the global split.
    
    Parameters:
    -----------
    config : DataConfiguration
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
    dataset_registry = DatasetRegistry(collection_name=config.dataset_registry_collection_name, directory_path=config.dataset_registry_directory)
    if config.use_dataset_registry:
        dataset_path = dataset_registry[(config, split_seed)]
    else:
        dataset_path = None
    if dataset_path is None:
        logging.info(f'Did not find precomputed dataset split.')
        if config.split_type == 'uniform':
            data_dict, fixed_vertices = load_data_from_configuration_uniform_split(config, split_seed)
        elif config.split_type == 'predefined':
            data_dict, fixed_vertices = load_data_from_configuration_predefined_split(config, split_seed)
        else:
            raise RuntimeError(f'Unsupported dataset split type {config.split_type}')
        if config.use_dataset_registry:
            dataset_registry[(config, split_seed)] = (data_dict, fixed_vertices)
    else:
        logging.info(f'Found precomputed dataset split at {dataset_path}')
        data_dict, fixed_vertices = torch.load(dataset_path)

    return data_dict, fixed_vertices
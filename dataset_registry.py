import seml
import os
import os.path as osp
import random
import shutil
from configuration import DataConfiguration
from typing import List, Tuple, Any
import logging
import datetime
import torch

DEFAULT_DATASET_REGISTRY_NAME = 'dataset_registry'

class DatasetRegistry:
    """ Dataset registry class that manages precomputed dataset splits.
    
    Parameters:
    -----------
    collection_name : str
        Which mongodb collection to use as regsitry.
    directory : str
        The directory in which to put the checkpoint files.
    """

    def __init__(self, collection_name: str =DEFAULT_DATASET_REGISTRY_NAME, 
                    directory_path: str ='/nfs/students/fuchsgru/dataset_registry'):
        self.database = seml.database
        self.collection_name = collection_name
        self.directory_path = directory_path 

    def _new_filename(self):
        """ Gets an unused random filename for the model registry. """
        keys = set(osp.splitext(p)[0] for p in os.listdir(self.directory_path))
        while True:
            new_key = str(random.randint(0, 1 << 32))
            if new_key not in keys:
                return new_key

    def items(self):
        """ Iterator for (cfg, path) pairs in the registry. """
        collection = self.database.get_collection(self.collection_name)
        for doc in collection.find():
            yield (DataConfiguration(**doc['config']), doc['seed']), doc['path']
        
    def __iter__(self):
        collection = self.database.get_collection(self.collection_name)
        return iter([(doc['config'], doc['seed']) for doc in collection.find()])

    def __len__(self):
        collection = self.database.get_collection(self.collection_name)
        return len([k for k in collection.find()])

    def __getitem__(self, key: Tuple[DataConfiguration, int]) -> str:
        """ Gets the path to a trained model with a given configuration set. """
        cfg, seed = key
        collection = self.database.get_collection(self.collection_name)
        for doc in collection.find():
            try:
                cfg_other = DataConfiguration(**doc['config'])
            except:
                # Older version of the registry may have attributes incompatible with current version
                continue
            if cfg_other.registry_configuration == cfg.registry_configuration and seed == doc['seed']:
                return doc['path']
        return None

    def remove(self, key: Tuple[DataConfiguration, int]) -> List[str]:
        """ Removes all experiment configurations from the registry that match."""
        cfg, seed = key
        ids_to_delete = []
        paths_to_delete = []
        collection = self.database.get_collection(self.collection_name)
        for doc in collection.find():
            try:
                cfg_doc = DataConfiguration(**doc['config'])
            except:
                # Older version of the registry may have attributes incompatible with current version
                continue
            if cfg.registry_configuration == cfg_doc.registry_configuration and seed == doc['seed']:
                ids_to_delete.append(doc['_id'])
                paths_to_delete.append(doc['path'])
        for _id in ids_to_delete:
            collection.delete_one({'_id' : _id})
        return paths_to_delete

    def __setitem__(self, key: Tuple[DataConfiguration, int], value: Any):
        """ Sets the path of a trained model with a given configuration set. """
        cfg, seed = key
        collection = self.database.get_collection(self.collection_name)
        paths_overwritten = self.remove((cfg, seed))
        new_path = osp.join(self.directory_path, self._new_filename() + '.pt')
        if len(paths_overwritten) > 0:
            logging.info(f'Paths {paths_overwritten} were overwritten with {new_path}')
        torch.save(value, new_path)
        collection.insert_one({
            'config' : cfg.registry_configuration,
            'path' : new_path,
            'time' : str(datetime.datetime.now()),
            'seed' : seed,
        })

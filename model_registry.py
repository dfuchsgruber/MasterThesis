import seml
import os
import os.path as osp
import random
import shutil
from configuration import ExperimentConfiguration, DEFAULT_REGISTRY_COLLECTION_NAME
from typing import List
import logging
import datetime

class ModelRegistry:
    """ Model registry class that saves and loads trained model checkpoints. 
    
    Parameters:
    -----------
    collection_name : str
        Which mongodb collection to use as regsitry.
    directory : str
        The directory in which to put the checkpoint files.
    copy_checkpoints_to_registry: bool
        If True, whenever a checkpoint is added to the registry, it is copied to `directory`.
    """

    def __init__(self, collection_name=DEFAULT_REGISTRY_COLLECTION_NAME, 
                    directory_path='/nfs/students/fuchsgru/model_registry', copy_checkpoints_to_registry=True):
        self.database = seml.database
        self.collection_name = collection_name
        self.directory_path = directory_path 
        self.copy_checkpoints_to_registry = copy_checkpoints_to_registry

    def _new_filename(self):
        """ Gets an unused random filename for the model registry. """
        keys = set(osp.splitext(p)[0] for p in os.listdir(self.directory_path))
        while True:
            new_key = str(random.randint(0, 1 << 32))
            if new_key not in keys:
                return new_key

    def _copy_checkpoint_to_new_file(self, cptk_path):
        """ Copies a checkpoint to a new file in the registry. """
        suffix = osp.splitext(cptk_path)[1]
        dst = osp.join(self.directory_path, self._new_filename() + suffix)
        shutil.copy2(cptk_path, dst)
        return dst

    def items(self):
        """ Iterator for (cfg, path) pairs in the registry. """
        collection = self.database.get_collection(self.collection_name)
        for doc in collection.find():
            yield ExperimentConfiguration(**doc['config']), doc['path']
        
    def __iter__(self):
        collection = self.database.get_collection(self.collection_name)
        return iter([doc['config'] for doc in collection.find()])

    def __len__(self):
        collection = self.database.get_collection(self.collection_name)
        return len([k for k in collection.find()])

    def __getitem__(self, cfg: ExperimentConfiguration):
        """ Gets the path to a trained model with a given configuration set. """
        collection = self.database.get_collection(self.collection_name)
        for doc in collection.find():
            if ExperimentConfiguration(**doc['config']).registry_configuration == cfg.registry_configuration:
                return doc['path']
        return None

    def remove(self, cfg: ExperimentConfiguration) -> List[str]:
        """ Removes all experiment configurations from the registry that match."""
        ids_to_delete = []
        paths_to_delete = []
        collection = self.database.get_collection(self.collection_name)
        for doc in collection.find():
            if cfg.registry_configuration == ExperimentConfiguration(**doc['config']).registry_configuration:
                ids_to_delete.append(doc['_id'])
                paths_to_delete.append(doc['path'])
        for _id in ids_to_delete:
            collection.delete_one({'_id' : _id})
        return paths_to_delete

    def __setitem__(self, cfg: ExperimentConfiguration, cptk_path):
        """ Sets the path of a trained model with a given configuration set. """
        collection = self.database.get_collection(self.collection_name)
        paths_overwritten = self.remove(cfg)
        if len(paths_overwritten) > 0:
            logging.info(f'Paths {paths_overwritten} were overwritten with {cptk_path}')
        if self.copy_checkpoints_to_registry:
            cptk_path = self._copy_checkpoint_to_new_file(cptk_path)
        collection.insert_one({
            'config' : cfg.registry_configuration,
            'path' : cptk_path,
            'time' : str(datetime.datetime.now()),
        })

import seml
import os
import os.path as osp
import random
import shutil
from configuration import ExperimentConfiguration, DEFAULT_REGISTRY_COLLECTION_NAME
from typing import List
import logging
import datetime
import json
import numpy as np
from time import time
from collections import defaultdict
from tqdm import tqdm

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

    def _new_filename(self, cfg_hash: int):
        """ Gets an unused random filename for the model registry. """
        rng = np.random.RandomState(seed=(cfg_hash & 0xFFFFFFFF))
        while True:
            keys = set(osp.splitext(p)[0] for p in os.listdir(self.directory_path))
            new_key = '-'.join([str(cfg_hash), str(time()).replace('.', '-'), str(rng.randint(0, 1 << 32))])
            if new_key not in keys:
                return new_key

    def _copy_checkpoint_to_new_file(self, cptk_path, cfg_hash: int):
        """ Copies a checkpoint to a new file in the registry. """
        suffix = osp.splitext(cptk_path)[1]
        dst = osp.join(self.directory_path, self._new_filename(cfg_hash) + suffix)
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
            try:
                cfg_other = ExperimentConfiguration(**doc['config'])
            except:
                # Older version of the registry may have attributes incompatible with current version
                continue
            if cfg_other.registry_configuration == cfg.registry_configuration:
                return doc['path']
        return None

    def remove(self, cfg: ExperimentConfiguration) -> List[str]:
        """ Removes all experiment configurations from the registry that match."""
        ids_to_delete = []
        paths_to_delete = []
        collection = self.database.get_collection(self.collection_name)
        for doc in collection.find():
            try:
                cfg_doc = ExperimentConfiguration(**doc['config'])
            except:
                # Older version of the registry may have attributes incompatible with current version
                continue
            if cfg.registry_configuration == cfg_doc.registry_configuration:
                ids_to_delete.append(doc['_id'])
                paths_to_delete.append(doc['path'])
        for _id in ids_to_delete:
            collection.delete_one({'_id' : _id})
        return paths_to_delete

    def __setitem__(self, cfg: ExperimentConfiguration, cptk_path):
        """ Sets the path of a trained model with a given configuration set. """
        collection = self.database.get_collection(self.collection_name)
        paths_overwritten = self.remove(cfg)
        cfg_hash = hash(json.dumps(cfg.registry_configuration, sort_keys=True))
        cptk_path_src = cptk_path
        if self.copy_checkpoints_to_registry:
            cptk_path = self._copy_checkpoint_to_new_file(cptk_path, cfg_hash)
        if len(paths_overwritten) > 0:
            logging.info(f'Paths {paths_overwritten} were overwritten with {cptk_path} (copied from {cptk_path_src}).')
        collection.insert_one({
            'config' : cfg.registry_configuration,
            'path' : cptk_path,
            'time' : str(datetime.datetime.now()),
        })


    def remove_duplicates(self, dry_run=True):
        """ Removes paths that are referenced by multiple configurations. (no idea how that could happen)... """

        collection = self.database.get_collection(self.collection_name)
        path_to_idxs = defaultdict(list)
        for doc in collection.find():
            path_to_idxs[doc['path']].append(doc['_id'])
        num_removed = 0
        for path, idxs in path_to_idxs.items():
            if len(idxs) > 1:
                if not dry_run:
                    for idx in idxs:
                        collection.delete_one({'_id' : idx})
                print(f'Removed configuration with ids {list(idxs)}.')
                num_removed += len(idxs)
        print(f'Removed {num_removed} configurations pointing to duplicate paths. Dry-Run: {dry_run}')

    def clean_directory(self, dry_run=True):
        """ Removes all files in the checkpoint directory that are not referenced by the current model registry. CAUTION! """

        files = set(osp.join(self.directory_path, p) for p in os.listdir(self.directory_path))
        collection = self.database.get_collection(self.collection_name)
        in_registry = set(doc['path'] for doc in collection.find())
            
        print(f'Number of files in directory {len(files)}.\nNumber of files in registry {len(in_registry)}.')
        num_deleted = 0
        for file in tqdm([f for f in files if f not in in_registry], desc=f'Removing checkpoints from {self.directory_path}'):
            if not dry_run:
                os.remove(file)
            # print(f'Removed file {file}')
            num_deleted += 1
        print(f'Deleted {num_deleted} of {len(files)} files.')
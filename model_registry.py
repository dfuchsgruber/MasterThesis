from util import dict_to_tuple
import seml
from copy import deepcopy
import os
import os.path as osp
import random
import shutil
from configuration import get_experiment_configuration

COLLECTION = '_model_registry'

class ModelRegistry:
    """ Model registry class that saves and loads trained model checkpoints. 
    
    Parameters:
    -----------
    collection_name : str
        Which mongodb collection to use as regsitry.
    keys_to_ignore : list
        A list of keys that is removed from configurations before matching with the registry.
        These should be keys in the configuration that don't affect the actual model.
        To access nested dicts, use the dot-notation.
    directory : str
        The directory in which to put the checkpoint files.
    copy_checkpoints_to_registry: bool
        If True, whenever a checkpoint is added to the registry, it is copied to `directory`.
    """

    def __init__(self, collection_name=COLLECTION, 
                    keys_to_ignore=[
                        'model.num_samples', 
                        'evaluation', 
                        'training.gpus',
                        ],
                    directory_path='/nfs/students/fuchsgru/model_registry', copy_checkpoints_to_registry=True):
        self.database = seml.database
        self.collection_name = collection_name
        self.keys_to_ignore = keys_to_ignore
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

    def _sanitze_config(self, base):
        """ Removes the dict keys that are ignored by the model registry from a configuration. """
        # It is esential, that also the configurations stored in the database are only considered as updates to the defaults.
        # This way, if new default parameters are added, the configurations in the database still can be matched to new experiments.
        return get_experiment_configuration(base, keys_to_ignore = self.keys_to_ignore)

    def items(self):
        """ Iterator for (cfg, path) pairs in the registry. """
        collection = self.database.get_collection(self.collection_name)
        return [(doc['config'], doc['path']) for doc in collection.find()]
        
    def __iter__(self):
        collection = self.database.get_collection(self.collection_name)
        return iter([doc['config'] for doc in collection.find()])

    def __len__(self):
        collection = self.database.get_collection(self.collection_name)
        return len([k for k in collection.find()])

    def __getitem__(self, cfg):
        """ Gets the path to a trained model with a given configuration set. """
        collection = self.database.get_collection(self.collection_name)
        for doc in collection.find():
            if self._sanitze_config(doc['config']) == self._sanitze_config(cfg):
                return doc['path']
        return None

    def __setitem__(self, cfg, cptk_path):
        """ Sets the path of a trained model with a given configuration set. """
        collection = self.database.get_collection(self.collection_name)
        if collection.delete_many({'config' : self._sanitze_config(cfg)}).deleted_count > 0:
            print(f'Overwriting path for configuration.')
        if self.copy_checkpoints_to_registry:
            cptk_path = self._copy_checkpoint_to_new_file(cptk_path)
        collection.insert_one({
            'config' : cfg,
            'path' : cptk_path,
        })

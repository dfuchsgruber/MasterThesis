from copy import deepcopy
from collections.abc import Mapping
import data.constants as dconstants

# Increment this number whenever you changed something essential that makes previously trained models invalid
REGISTRY_VERSION = 0 

default_configuration = {
    'model' : {
        'hidden_sizes' : [64,],
        'weight_scale' : 1.0,
        'use_spectral_norm' : False,
        'num_heads' : -1,
        'diffusion_iterations' : 5,
        'teleportation_probability' : 0.1,
        'model_type' : 'gcn',
        'use_bias' : True,
        'activation' : 'leaky_relu',
        'leaky_relu_slope' : 0.01,
        'normalize' : True,
        'residual' : False,
        'freeze_residual_projection' : False,
        'num_ensemble_members' : 1,
        'num_samples' : 1,
        'dropout' : 0.0,
        'drop_edge' : 0.0,
        'use_spectral_norm_on_last_layer' : True,
        'cached' : False,
        'self_loop_fill_value' : 1.0,
    },
    'data' : {
        'dataset' : 'cora_ml',
        'num_dataset_splits' : 1,
        'train_portion' : 20,
        'val_portion' : 20,
        # 'test_portion' : 0.6,
        'test_portion_fixed' : 0.2,
        'train_labels' : 'all',
        'perturbation_budget' : 0.1,
        'setting' : dconstants.HYBRID[0],
        'ood_type' : dconstants.LEFT_OUT_CLASSES[0],
        'left_out_class_labels' : 'all',
        'ood_sampling_strategy' : dconstants.SAMPLE_ALL[0],
        # 'train_labels_remove_other' : True,
        # 'val_labels' : 'all',
        # 'val_labels_remove_other' : True,
        'split_type' : 'uniform',
        'base_labels' : 'all',
        'type' : 'gust',
        'corpus_labels' : 'all',
        'min_token_frequency' : 10,
        "preprocessing" : "bag_of_words",
        "language_model" : "bert-base-uncased",
        'drop_train_vertices_portion' : 0.0,
        'normalize' : 'l2',
        'vectorizer' : 'tf-idf',
    },
    'training' : {
        'max_epochs' : 1000,
        'learning_rate' : 0.01,
        'early_stopping' : {
            'patience' : 100,
            'mode' : 'min',
            'monitor' : 'val_loss',
            'min_delta' : 1e-3,
        },
        'gpus' : 1,
    },
    'evaluation' : {
        'pipeline' : [],
        'print_pipeline' : True,
        'ignore_exceptions' : False,
    },
    'version' : REGISTRY_VERSION,
}

def remove_from_configuration(cfg, key, not_exists_ok=True):
    """ Removes a key from a configuration. 
    
    Parameters:
    -----------
    cfg : dict
        The configuration to update
    key : str
        The key (in dot-notation) to remove.
    not_exists_ok : bool
        If it is ok if the key does not exist in cfg. Otherwise, an Exception will be raised.
    """
    path = key.split('.')
    x = cfg
    while len(path) > 1:
        if not isinstance(x, Mapping):
            raise RuntimeError(f'Can not index level {path[0]} of key {key} because it is type {type(x)}.')
        if path[0] in x:
            x = x[path[0]]
            path = path[1:]
        else:
            if not_exists_ok:
                return cfg
            else:
                raise RuntimeError(f'Key {path[0]} does not exist in configuration.')
    assert len(path) == 1, f'{len(path)}'
    if not isinstance(x, Mapping):
        raise RuntimeError(f'Can not index level {path[0]} of key {key} because it is type {type(x)}.')
    if path[0] in x:
        x.pop(path[0])
        return cfg
    else:
        if not_exists_ok:
            return cfg
        else:
            raise RuntimeError(f'Key {path[0]} does not exist in configuration.')


def update_configuration(default_cfg, update_cfg, keys_to_ignore):
    """ Updates a configuration. 
    
    Parameteres:
    ------------
    default_cfg : dict
        The base configuration.
    update_cfg : dict
        The updates.
    keys_to_ignore:
        Keys in `update_cfg` that will not be updated.
        Also, these keys will not be part of the final configuration.
        Use dot-notation to access nested dicts.
    """
    default_cfg = deepcopy(default_cfg)
    update_cfg = deepcopy(update_cfg)

    # Remove update keys
    for key in keys_to_ignore:
        remove_from_configuration(update_cfg, key)
        remove_from_configuration(default_cfg, key)
    
    def _merge(a, b):
        for key in b.keys():
            if key not in a:
                a[key] = b[key]
            else:
                if isinstance(a[key], Mapping) and isinstance(b[key], Mapping):
                    _merge(a[key], b[key])
                else:
                    a[key] = b[key]
    
    _merge(default_cfg, update_cfg)
    return default_cfg

def get_experiment_configuration(cfg, keys_to_ignore=[]):
    return update_configuration(default_configuration, cfg, keys_to_ignore=keys_to_ignore)
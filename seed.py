# Module to manage global seeds for the project

import numpy as np
import zlib

_SEED_UPPER_BOUND = 0x100000000 # Upper bound for seeds
DATA_SPLIT_SEED = 1337 # Seed that affects data splitting
MODEL_INIT_SEED = 1337 # Seed that affects model initialization

class DataSplitSeedsIterator:
    """ Infinite iterator for data split seeds. """
    def __init__(self, seed=DATA_SPLIT_SEED):
        self.seed = seed
    
    def __next__(self):
        return self.rng.randint(0, _SEED_UPPER_BOUND)

    def __iter__(self):
        self.rng = np.random.RandomState(self.seed)
        return self

def data_split_seeds_iterator():
    """ Returns an infinite iterator for data split seeds """
    return iter(DataSplitSeedsIterator())

def data_split_seeds(num):
    """ Returns `num` seed for data splitting.
    
    Parameters:
    -----------
    num : int
        How many seeds to generate.
    
    Returns:
    --------
    seeds : ndarray, shape [num]
        Random seeds.
    """
    rng = np.random.RandomState(DATA_SPLIT_SEED)
    return rng.randint(0, _SEED_UPPER_BOUND, num)

def model_seeds(num, model_name=None):
    """ Returns `num` seed for model initialization.
    
    Parameters:
    -----------
    num : int
        How many seeds to generate.
    model_name : object
        The name of the model to initialize. This ensures that different models get different seed sets.
    
    Returns:
    --------
    seeds : ndarray, shape [num]
        Random seeds.
    """
    seed = zlib.adler32(bytes(f'{model_name}:{MODEL_INIT_SEED}', 'utf-8')) % (1 << 32)
    rng = np.random.RandomState(seed)
    return rng.randint(0, _SEED_UPPER_BOUND, num)

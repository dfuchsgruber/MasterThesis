# Module to manage global seeds for the project

import numpy as np
import zlib

_SEED_UPPER_BOUND = 0x100000000 # Upper bound for seeds
DATA_SPLIT_SEED = 1337 # Seed that affects data splitting
MODEL_INIT_SEED = 42 # Seed that affects model initialization

class SeedsIterator:
    """ Infinite iterator for seeds. """
    def __init__(self, seed):
        self.seed = seed
    
    def __next__(self):
        return self.rng.randint(0, _SEED_UPPER_BOUND)

    def __iter__(self):
        self.rng = np.random.RandomState(self.seed)
        return self

def data_split_seeds_iterator():
    """ Returns an infinite iterator for data split seeds """
    return iter(SeedsIterator(DATA_SPLIT_SEED))

def model_seeds_iterator():
    """ Returns an infinite iterator for model seeds. """
    return iter(SeedsIterator(MODEL_INIT_SEED))

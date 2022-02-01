# Module to manage global seeds for the project

import numpy as np
import zlib

_SEED_UPPER_BOUND = 0x100000000 # Upper bound for seeds
DATA_SPLIT_SEED = 1338 # Seed that affects data splitting
DATA_SPLIT_FIXED_TEST_SEED = 1125387415 # Seed that is used to determine fixed test portions
MODEL_INIT_SEED = 42 # Seed that affects model initialization

class SeedIterator:
    """ Infinite iterator for seeds. """
    def __init__(self, seed):
        self.seed = seed
    
    def __next__(self):
        return self.rng.randint(0, _SEED_UPPER_BOUND)

    def __iter__(self):
        self.rng = np.random.RandomState(self.seed)
        return self

class Seeds:
    """ Class to retrieve seeds. """

    def __init__(self, seed: int):
        self._iterator = iter(SeedIterator(seed))
        self._cache = []

    def __getitem__(self, idx: int) -> int:
        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f'Can only retrieve seeds from a positive index not {idx}')
        while len(self._cache) <= idx:
            self._cache.append(next(self._iterator))
        return self._cache[idx]

def data_split_seeds() -> Seeds:
    """ Returns an indexable seeds class for data split seeds """
    return Seeds(DATA_SPLIT_SEED)

def model_seeds() -> Seeds:
    """ Returns an indexable seeds class for model seeds. """
    return Seeds(MODEL_INIT_SEED)


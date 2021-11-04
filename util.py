from contextlib import contextmanager
import os, sys
import torch
import numpy as np

@contextmanager
def suppress_stdout(supress=True):
    """ From: https://stackoverflow.com/a/25061573 """
    if supress:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:  
                yield
            finally:
                sys.stdout = old_stdout
    else:
        try:
            yield
        finally:
            pass

def is_outlier(array, quantile=0.95):
    """ Detects outliers as samples that are not within a certain quantile of the data. 
    
    Parameters:
    -----------
    array : ndarray, shape [N]
        The array to find outliers in.
    quantile : float
        How much of the data to include.

    Returns:
    --------
    outliers : ndarray, shape [N]
        Array that masks all outliers, i.e. `outliers[i] == True` if a point is identified as outlier.    
    """
    finite = np.isfinite(array)
    array_finite = array[finite]
    idxs = np.argsort(array_finite)
    delta = (1 - quantile) / 2
    upper, lower = int((quantile + delta) * array_finite.shape[0]), int(delta * array_finite.shape[0])
    idxs = idxs[lower:upper]
    # Translate to the whole array
    idxs = np.arange(array.shape[0])[finite][idxs]
    is_outlier = np.ones_like(array, dtype=bool)
    is_outlier[idxs] = False
    return is_outlier

def random_derangement(N, rng=None):
    """ Returns a random permutation of [0, 1, ..., N] that has no fixed points (derangement).
    
    Parameters:
    -----------
    N : int
        The number of elements to derange.
    rng : np.random.RandomState or None
        The random state to use. If None is given, a random RandomState is generated.

    References:
    -----------
    Taken from: https://stackoverflow.com/a/52614780
    """
    if rng is None:
        rng = np.random.RandomState(np.random.randint(1 << 32))
    
    # Edge cases
    if N == 0:
        return np.array([], dtype=int)
    elif N == 1:
        return np.array([0], dtype=int)

    original = np.arange(N)
    new = rng.permutation(N)
    same = np.where(original == new)[0]
    _iter = 0
    while len(same) != 0:
        if _iter + 1 % 100 == 0:
            print('iteration', _iter)
        swap = same[rng.permutation(len(same))]
        new[same] = new[swap]
        same = np.where(original == new)[0]
        if len(same) == 1:
            swap = rng.randint(0, N)
            new[[same[0], swap]] = new[[swap, same[0]]]
    return new
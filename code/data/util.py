import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

GLOBAL_SPLIT_SEED = 1337

def stratified_split(ys, num_splits, sizes=[0.05, 0.15, 0.8], random_state=GLOBAL_SPLIT_SEED):
    """ Performs several splits of a dataset into index sets using a stratified strategy.
    
    Parameters:
    -----------
    ys : ndarray, shape [N]
        Labels that are used for a stratified split.
    num_splits : int
        How many splits are calculated.
    sizes : sequence of ints
        Size of each split. Should sum up to 1.0
    random_state : int or None
        If given, seed for generating all splits to ensure reproducability.

    Returns:
    --------
    index_sets : list or ndarrays, shape [num_splits, index_set_size]
        All index sets of different size.
    """
    assert np.allclose(sum(sizes), 1.0), f'Sizes should sum up to 1.0'
    if random_state is not None:
        np.random.seed(random_state)
        
    sizes = np.array(sizes)
    index_sets = [[] for _ in sizes]
    for _ in range(num_splits):
        partition = [[] for _ in sizes]
        for v, count in zip(*np.unique(ys, return_counts=True)):
            idxs = np.where(ys == v)[0]
            np.random.shuffle(idxs)
            
            # Partition this subset
            start_idx = 0
            for set_idx, endpoint in enumerate(np.cumsum(sizes)):
                end_idx = int(len(idxs) * endpoint)
                partition[set_idx].append(idxs[start_idx : end_idx])
                start_idx = end_idx
        
        for set_idx, idxs in enumerate(partition):
            index_sets[set_idx].append(np.concatenate(idxs))
    
    return [np.array(index_set) for index_set in index_sets]

if __name__ == '__main__':
    ys = np.array([0] * 100 + [1] * 150 + [2] * 250)
    a, b, c = stratified_split(ys, 3, sizes=[0.3, 0.3, 0.4])
    print(a.shape, b.shape, c.shape)
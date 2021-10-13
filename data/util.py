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
    mask : list or ndarrays, shape [sizes, num_splits, N]
        Masks for all sets.
    """
    assert np.allclose(sum(sizes), 1.0), f'Sizes should sum up to 1.0'
    if random_state is not None:
        np.random.seed(random_state)
        
    sizes = np.array(sizes)
    mask = np.zeros((len(sizes), num_splits, len(ys)), dtype=bool)
    for split_idx in range(num_splits):
        partition = [[] for _ in sizes]
        for v, count in zip(*np.unique(ys, return_counts=True)):
            idxs = np.where(ys == v)[0]
            np.random.shuffle(idxs)
            
            # Partition this subset
            start_idx = 0
            for set_idx, endpoint in enumerate(np.cumsum(sizes)):
                end_idx = int(len(idxs) * endpoint)
                mask[set_idx, split_idx][idxs[start_idx : end_idx]] = True
                start_idx = end_idx

    return mask

def data_get_num_attributes(data):
    return data[0].x.size(1)

def data_get_num_classes(data):
    return int((data[0].y.max() + 1).item())

if __name__ == '__main__':
    ys = np.array([0] * 100 + [1] * 150 + [2] * 250)
    mask = stratified_split(ys, 4, sizes=[0.3, 0.3, 0.4])
    print(mask.shape)
    print((mask.sum(0) == 1).all())
import torch
import torch.nn.functional as F
import numpy as np
from util import get_k_hop_neighbourhood
import evaluation.callbacks

def split_labels_into_id_and_ood(y, id_labels, ood_labels=None, id_label=0, ood_label=1):
    """ Creates new labels that correspond to in-distribution and out-ouf-distribution.
    
    Paramters:
    ----------
    y : torch.tensor, shape [N]
        Original class labels.
    id_labels : iterable
        All in-distribution labels.
    ood_labels : iterable
        All out-of-distribution labels or None. If None, it will be set to all labels in y that are not id.
    id_label : int
        The new label of id points.
    ood_label : int
        The new label of ood points.

    Returns:
    --------
    y_new : torch.tensor, shape [N]
        Labels corresponding to id and ood data.
    """
    if ood_labels is None:
        ood_labels = set(torch.unique(y).tolist()) - set(id_labels)

    y_new = torch.zeros_like(y)
    for label in id_labels:
        y_new[y == label] = id_label
    for label in ood_labels:
        y_new[y == label] = ood_label
    return y_new

def get_data_loader(name, loaders):
    """ Gets the right dataloader given the name of a dataset. """
    name = name.lower()
    if name not in loaders:
        raise RuntimeError(f'Cant provide dataset {name} to evaluation.')
    else:
        return loaders[name]

def run_model_on_datasets(model, data_loaders, gpus=0, callbacks=[
    evaluation.callbacks.make_callback_get_features(), 
    evaluation.callbacks.make_callback_get_predictions(), 
    evaluation.callbacks.make_callback_get_ground_truth()
], run_model=True):
    """ Extracts features of all data loaders. 
    
    model : torch.nn.Module
        A model to use as a feature extractor
    data_loaders : list
        A list of data loaders to extract features for.
    gpus : int
        If > 0, models are run on the gpu.
    callbacks : list
        A list of functions that is run on every pair of (data, model_output). Its results are aggregated into lists.
    run_model : bool
        If `False`, the model will not be run at all and `model_output` will be `None`.

    Returns:
    --------
    results : tuple
        Results for each callback. Each result is a list per data loader.
    """
    if gpus > 0:
        model = model.to('cuda')
    
    results = tuple([] for _ in callbacks)
    for loader in data_loaders:
        assert len(loader) == 1, f'Feature extraction is currently only supported for single graphs.'
        for data in loader:
            if gpus > 0:
                data = data.to('cuda')
            if run_model:
                output = model(data)
            else:
                output = None
            for idx, callback in enumerate(callbacks):
                results[idx].append(callback(data, output))
    return results


def count_neighbours_with_label(data_loader, labels, k=1, mask=True):
    """ For each vertex in a dataset, counts the number of neighbours that have a certain label.  
    
    Parameters:
    -----------
    data_loader : torch_geometric.data.DataLoader
        The data loader instance. Only supports single graphs.
    label : int
        The labels to search for.
    k : int
        How many hops to look ahead. This is exact, i.e. if k=2, then direct neighbours will not be considered.
    mask : bool
        If True, only vertices within the graph's veretx mask are considered.

    Returns:
    --------
    torch.Tensor, shape [N_graph]
        Counts how many neighbours with a label in `labels` each vertex has in its k-hop neighbourhood.
    torch.Tensor, shape [N_graph]
        Counts how many neighbours a vertex has in its k-hop neighbourhood regardless of its label.
    """
    assert len(data_loader.dataset) == 1, f'Data loader loads more than 1 sample.'
    data = data_loader.dataset[0].cpu()
    if labels == 'all':
        labels = set(torch.unique(data.y).cpu().tolist())
    label_mask = split_labels_into_id_and_ood(data.y, labels, id_label=1, ood_label=0).bool()
    neighbours = get_k_hop_neighbourhood(data.edge_index, k, k_min=k)
    count, total = torch.tensor([
        label_mask[np.array(neighbours.get(idx, []))].sum() for idx in range(data.x.size(0))
    ]).long(), torch.tensor([
        len(neighbours.get(idx, [])) for idx in range(data.x.size(0))
    ]).long() 
    if mask:
        count, total = count[data.mask], total[data.mask]
    return count, total

def get_out_of_distribution_labels(data_loaders, k_max, train_labels, mask=True):
    """ Gets information about which vertices are from an out-of-distribution class and how many of those are in neighbourhoods.
    
    Parameters:
    -----------
    data_loaders : list
        A list of data-loaders to get the information of.
    k_max : int
        The number of neighbours with an out-of-distribution class will be returned for k-hop neighbourhoods where k is in [1, 2, ... k_max]
    train_labels : set
        All labels that are considered in-distribution.
    mask : bool
        If the information is to be extracted for only the vertices in a datasets mask or all vertices in the graph.

    Returns:
    --------
    mask_is_train_label : list
        For each loader, a mask that indicates if a vertex has a label within the train label set.
    num_train_label_nbs : list
        `k_max` lists, one for each k-hop neighbourhood. Elements in these lists are, for each loader, a tensor that that counts how many of a vertex's neighbours in a k-hop neighbourhood have an out of distribution label.
    num_nbs : list
        `k_max` lists, one for each k-hop neighbourhood. Elements in these lists are, for each loader, a tensor that gives the size of the k-hop neighbourhood of a vertex.
    """
    train_labels = set(train_labels)
    callbacks = [
        evaluation.callbacks.make_callback_is_ground_truth_in_labels(train_labels, mask=True)
    ]
    for k in range(1, k_max + 1):
        callbacks += [
            evaluation.callbacks.make_callback_count_neighbours_with_labels(train_labels, k, mask=True),
            evaluation.callbacks.make_callback_count_neighbours(k, mask=True)
        ]
    results = run_model_on_datasets(None, data_loaders, callbacks=callbacks, run_model=False)

    return results[0], results[1::2], results[2::2]
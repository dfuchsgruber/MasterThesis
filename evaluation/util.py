import torch
import torch.nn.functional as F
import numpy as np
from util import get_k_hop_neighbourhood
import evaluation.callbacks
import evaluation.constants as evaluation_constants
from collections.abc import Iterable

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
    evaluation.callbacks.make_callback_get_ground_truth(),
], run_model=True, model_kwargs={}):
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
    model_kwargs : dict
        Keyword arguments that are passed to the model.

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
                output = model(data, **model_kwargs)
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


def get_distribution_labels_perturbations(data_loaders):
    """ Gets information about which vertices are perturbed.
    
    Parameters:
    -----------
    data_loaders : list
        A list of data-loaders to get the information of.

    Returns:
    --------
    distribution_labels : torch.Tensor, [N]
        Each vertex in the dataset with its distribution label (according to `evaluation.constants`)
    """
    callbacks = [
        evaluation.callbacks.make_callback_get_perturbation_mask(mask=True, cpu=True),
    ]
    results = run_model_on_datasets(None, data_loaders, callbacks=callbacks, run_model=False)
    is_perturbed = torch.cat(results[0], dim=0)
    distribution_labels = torch.empty_like(is_perturbed).long()
    distribution_labels[~is_perturbed] = evaluation.constants.ID_CLASS_NO_OOD_CLASS_NBS
    distribution_labels[is_perturbed] = evaluation.constants.OOD_CLASS_NO_ID_CLASS_NBS
    return distribution_labels

def get_distribution_labels_leave_out_classes(data_loaders, k_max, train_labels, mask=True, threshold=0.0):
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
    threshold : float or list
        The fraction of neighbours that is ignored for the labeling. For example, if a vertex with an in-distribution class
        has less than theshold out-of-distribution-class neighbours, it will still be labeled `in distribution with only in distribution nbs`
        If a single value is given, then the cirterion for purity within the k-hop neighbourhood will be set to (1 - threshold)**k.
        If a list is given, then this purity threshold within the k-hop neighbourhood will be 1-threshold[k - 1].

        Use a value of `0.0` for a strict labeling.

    Returns:
    --------
    distribution_labels : torch.Tensor, [N]
        Each vertex in the dataset with its distribution label (according to `evaluation.constants`)
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

    mask_is_train_label, num_train_label_nbs, num_nbs = results[0], results[1::2], results[2::2]

    # Concatenate values over all datasets
    mask_is_train_label = torch.cat(mask_is_train_label, dim=0) # N
    num_train_label_nbs = torch.stack([torch.cat(l, dim=0) for l in num_train_label_nbs]) # k, N

    # Calculate purity of each k-hop neighbourhood
    num_nbs = torch.stack([torch.cat(l, dim=0) for l in num_nbs]) # k, N
    fraction_train_label_nbs = (num_train_label_nbs / num_nbs).permute(1, 0) # N, k
    if isinstance(threshold, Iterable):
        threshold = 1 - np.array(threshold)
    else:
        threshold = np.array([(1 - threshold)**k for k in range(1, k_max + 1)])
    threshold = torch.tensor(threshold)
    mask_has_only_train_label_nbs = (fraction_train_label_nbs >= threshold).all(dim=1) # N
    mask_has_only_non_train_label_nbs = ((1 - fraction_train_label_nbs) >= threshold).all(dim=1) # N

    # Plot the softmax-entropy distribution by train-label data i) with no non-train label neighbours ii) with at least one train label neighbour
    distribution_labels = torch.empty_like(mask_is_train_label).long()
    distribution_labels[(~mask_is_train_label) & (mask_has_only_non_train_label_nbs)] = evaluation_constants.OOD_CLASS_NO_ID_CLASS_NBS
    distribution_labels[(~mask_is_train_label) & (~mask_has_only_non_train_label_nbs)] = evaluation_constants.OOD_CLASS_ID_CLASS_NBS
    distribution_labels[mask_is_train_label & (mask_has_only_train_label_nbs)] = evaluation_constants.ID_CLASS_NO_OOD_CLASS_NBS
    distribution_labels[mask_is_train_label & (~mask_has_only_train_label_nbs)] = evaluation_constants.ID_CLASS_ODD_CLASS_NBS

    # print(torch.unique(distribution_labels, return_counts=True), train_labels)

    return distribution_labels

def separate_distributions_leave_out_classes(distribution_labels, separate_distributions_by):
    """ Gets the labels and mask used for auroc computations by separating the two data distributions.
    
    Parameters:
    -----------
    distribution_labels : torch.Tensor, shape [N]
        Distribution labels according to `get_distribution_labels`.
    separate_distributions_by : str
        With which criterion to select and separate vertices used for the auroc computation.
        Possible are:
            - 'train-labels' : All vertices are used for auroc computation and positives are vertices with train labels.
            - 'train-lables-neighbour' : Only vertices with train labels and neighbours within train-labels as well as vertices with non-train labels and neighbours within non-train labels are selected. Positives are vertices with train labels.
    
    Returns:
    --------
    auroc_labels : torch.Tensor, shape [N]
        The labels used for auroc calculation.
    auroc_mask : torch.Tensor, shape [N]
        A mask that only selects vertices used for auroc calculation.
    """
    is_id_class = torch.zeros_like(distribution_labels).bool()
    for label in evaluation_constants.ID_CLASS:
        is_id_class |= (distribution_labels == label)

    if separate_distributions_by.lower() in ('train_label', 'train-label'):
        auroc_mask = torch.ones_like(is_id_class) # All vertices are used for the auroc
        auroc_labels = is_id_class # Positive vertices are those that hold a train label
    elif separate_distributions_by.lower() in ('neighbourhood', 'neighborhood'):
        # For auroc, only use vertices with train label & exclusively train label nbs as well as non-train-label and exclusively non-train-label-nbs
        auroc_mask = (distribution_labels == evaluation_constants.ID_CLASS_NO_OOD_CLASS_NBS) | (distribution_labels == evaluation_constants.OOD_CLASS_NO_ID_CLASS_NBS)
        auroc_labels = is_id_class # Positive vertices are those that hold a train label
    else:
        raise RuntimeError(f'Unknown distribution separation {separate_distributions_by}')

    return auroc_labels, auroc_mask
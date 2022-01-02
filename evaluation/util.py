import torch
import torch.nn.functional as F
import numpy as np
from util import get_k_hop_neighbourhood
import evaluation.callbacks
import evaluation.constants as evaluation_constants
from collections.abc import Iterable

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


def get_distribution_labels(data_loaders, k_max, mask=True, threshold=0.0):
    """ Gets information about which vertices have the attribute `is_out_of_distribution`. It counts for each vertex how big the fraction
    of vertices with that attribute is in its k-neighbourhood. For a k-neighbourhood, a threshold `t_k = 1 - (threshold ** k)` is defined.
    
    A vertex will be assigned the following labels:
        - evaluation_constants.OOD_CLASS_NO_ID_CLASS_NBS : If a vertex has the `is_out_of_distribution` attribute and for all k-neighbourhoods
            the fraction of vertices with the same attribute is >= t_k
        - evaluation_constants.OOD_CLASS_ID_CLASS_NBS : If a vertex has the `is_out_of_distribution` and for at least one k-neighbourhood
            the fraction of vertices with the same attribute is < t_k
        - evaluation_constants.ID_CLASS_NO_OOD_CLASS_NBS : If a vertex doesn't have the `out_of_distribution` attribute and for all
            k-neighbourhoods the fraction of vertices without that attribute is >= t_k
        - evaluation_constants.ID_CLASS_NO_OOD_CLASS_NBS
    
    Parameters:
    -----------
    data_loaders : list
        A list of data-loaders to get the information of.
    k_max : int
        The number of neighbours with an out-of-distribution class will be returned for k-hop neighbourhoods where k is in [1, 2, ... k_max]
        Pass k_max = 0 to ignore neighbours.
    mask : bool
        If the information is to be extracted for only the vertices in a datasets mask or all vertices in the graph.
    threshold : float or list
        The fraction of neighbours that is ignored for the labeling. For example, if a vertex with an in-distribution attribute
        has less than theshold out-of-distribution-attribute neighbours, it will still be labeled `in distribution with only in distribution nbs`
        If a single value is given, then the cirterion for purity within the k-hop neighbourhood will be set to (1 - threshold)**k.
        If a list is given, then this purity threshold within the k-hop neighbourhood will be 1-threshold[k - 1].

        Use a value of `0.0` for a strict labeling.

    Returns:
    --------
    distribution_labels : torch.Tensor, [N]
        Each vertex in the dataset with its distribution label (according to `evaluation.constants`)
    """
    callbacks = [
        evaluation.callbacks.make_callback_count_neighbours_with_attribute(lambda data, output: ~data.is_out_of_distribution.numpy(), 0, mask=True)
    ]
    for k in range(1, k_max + 1):
        callbacks += [
            evaluation.callbacks.make_callback_count_neighbours_with_attribute(lambda data, output: ~data.is_out_of_distribution.numpy(), k, mask=True),
            evaluation.callbacks.make_callback_count_neighbours(k, mask=True)
        ]
    results = run_model_on_datasets(None, data_loaders, callbacks=callbacks, run_model=False)

    mask_is_id, num_id_nbs, num_nbs = results[0], results[1::2], results[2::2]

    # Concatenate values over all datasets
    mask_is_id = torch.cat(mask_is_id, dim=0) > 0 # N
    num_id_nbs = torch.stack([torch.cat(l, dim=0) for l in num_id_nbs]) # k, N

    # Calculate purity of each k-hop neighbourhood
    num_nbs = torch.stack([torch.cat(l, dim=0) for l in num_nbs]) # k, N
    fraction_id_nbs = (num_id_nbs / num_nbs).permute(1, 0) # N, k
    if isinstance(threshold, Iterable):
        threshold = 1 - np.array(threshold)
    else:
        threshold = np.array([(1 - threshold)**k for k in range(1, k_max + 1)])
    threshold = torch.tensor(threshold)
    mask_has_only_id_nbs = (fraction_id_nbs >= threshold).all(dim=1) # N
    mask_has_only_ood_nbs = ((1 - fraction_id_nbs) >= threshold).all(dim=1) # N

    # Plot the softmax-entropy distribution by train-label data i) with no non-train label neighbours ii) with at least one train label neighbour
    distribution_labels = torch.empty_like(mask_is_id).long()
    distribution_labels[(~mask_is_id) & (mask_has_only_ood_nbs)] = evaluation_constants.OOD_CLASS_NO_ID_CLASS_NBS
    distribution_labels[(~mask_is_id) & (~mask_has_only_ood_nbs)] = evaluation_constants.OOD_CLASS_ID_CLASS_NBS
    distribution_labels[mask_is_id & (mask_has_only_id_nbs)] = evaluation_constants.ID_CLASS_NO_OOD_CLASS_NBS
    distribution_labels[mask_is_id & (~mask_has_only_id_nbs)] = evaluation_constants.ID_CLASS_ODD_CLASS_NBS

    return distribution_labels

def separate_distributions(distribution_labels, separate_distributions_by):
    """ Gets the labels and mask used for auroc computations by separating the two data distributions.
    
    Parameters:
    -----------
    distribution_labels : torch.Tensor, shape [N]
        Distribution labels according to `get_distribution_labels`.
    separate_distributions_by : str
        With which criterion to select and separate vertices used for the auroc computation.
        Possible are:
            - 'ood' : All vertices are used for auroc computation and positives are the one without the attribute `is_out_of_distribution`.
            - 'ood-and-neighbours' : Only vertices i) with the attribute `is_out_of_distribution` and all k-hops neighbours with the same attribute
                (up to a threshold) and ii) without the attribute `is_out_of_distribution` and all k-hop neighbours without the same attribute
                (up to a threshold) are considered for auroc computation. Positives are the ones without the attriubte `is_out_of_distribution`
    
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

    if separate_distributions_by.lower() in ('ood',):
        auroc_mask = torch.ones_like(is_id_class) # All vertices are used for the auroc
        auroc_labels = is_id_class # Positive vertices are those that hold a train label
    elif separate_distributions_by.lower() in ('ood-and-neighbours', 'ood-and-neighbourhood'):
        # For auroc, only use vertices with train label & exclusively train label nbs as well as non-train-label and exclusively non-train-label-nbs
        auroc_mask = (distribution_labels == evaluation_constants.ID_CLASS_NO_OOD_CLASS_NBS) | (distribution_labels == evaluation_constants.OOD_CLASS_NO_ID_CLASS_NBS)
        auroc_labels = is_id_class # Positive vertices are those that hold a train label
    else:
        raise RuntimeError(f'Unknown distribution separation {separate_distributions_by}')

    return auroc_labels, auroc_mask
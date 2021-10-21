import numpy as np
import torch
from torch_geometric.data import Data

def local_lipschitz_bounds(perturbations):
    """ Function that calculates local Lipschitz bounds given some random perturbations. 
    
    Parameters:
    -----------
    perturbations : dict
        Mapping from perturbation_magnitude -> tensor of output perturbations.

    Returns:
    --------
    slope_mean : float
        The slope of the linear function fit to mean perturbations.
    slope_median : float
        The slope of the linear function fit to median perturbations.
    slope_max : float
        The slope of the linear function that upper bounds all perturbations. This is an empirical (tight) bound on the local Lipschitz constant.
    slope_min : float
        The slope of the linear function that lower bounds all perturbations.
    """ 
    xs = torch.tensor(list(perturbations.keys())).numpy()
    ys = torch.stack(list(perturbations.values()))
    means, meds, maxs, mins = ys.mean(dim=-1), ys.median(dim=-1)[0].numpy(), ys.max(axis=-1)[0].numpy(), ys.min(axis=-1)[0].numpy()

    return (means / xs).mean(), (meds / xs).mean(), (maxs / xs).mean(), (mins / xs).mean()


@torch.no_grad()
def logit_space_bounds(model, dataset):
    """ Gets bounds for the logit space in each dimension given a certain dataset. 
    
    Parameters:
    -----------
    model : torch.nn.Module
        A torch model.
    dataset : torch_geometric.data.Dataset
        A dataset.
    
    Returns:
    --------
    min : torch.tensor, shape [num_classes]
        Minimal values in logit space.
    max : torch.tensor, shape [num_classes]
        Maximal values in logit space
    """
    logits = model(dataset[0])[-1][dataset[0].mask]
    mins, maxs = logits.min(dim=0)[0], logits.max(dim=0)[0]
    for idx in range(1, len(dataset)):
        logits = model(dataset[idx])[-1][dataset[idx].mask]
        mins = torch.min(mins, logits.min(dim=0)[0])
        maxs = torch.max(maxs, logits.max(dim=0)[0])
    return mins, maxs


@torch.no_grad()
def local_perturbations(model, dataset, perturbations=np.linspace(0.1, 5.0, 50), num_perturbations_per_sample=10, seed=None):
    """ Locally perturbs data and see how much the logits are perturbed. 
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to check
    dataset : torch_geometric.data.Dataset
        The dataset to investigate. Currently, only dataset[0] is used.
    perturbations : iterable
        Different magnitudes of noise to check.
    num_perturbations_per_sample : int
        How many random perturbations to use per sample and per noise magnitude.
    seed : int or None
        If given, seeds the rng and makes the perturbations deterministic.
    
    Returns:
    --------
    perturbations : dict
        Mapping from input_perturbations -> `len(dataset[0]) * num_perturbations_per_sample` output perturbations.
    """
    if seed is not None:
        torch.manual_seed(seed)
    logits = model(dataset[0])[-1]
    result = {}
    for eps in perturbations:
        results_eps = []
        for _ in range(num_perturbations_per_sample):
            noise = torch.randn(list(dataset[0].x.size()))
            noise = noise / noise.norm(dim=-1, keepdim=True) * eps
            x_perturbed = dataset[0].x + noise
            data = Data(x=x_perturbed, edge_index=dataset[0].edge_index)
            logits_perturbed = model(data)[-1]
            results_eps.append((logits - logits_perturbed).norm(dim=1))
            
        result[eps] = torch.cat(results_eps)
    return result
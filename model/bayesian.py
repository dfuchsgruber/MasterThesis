import torch.nn as nn
import torch
import torch_geometric
from torch import Tensor
import numpy as np
from typing import Union

def sample_normal(mean: Tensor, logsigma: Tensor):
    """ Draws a sample from a normal distribution using reparametrization. 
    
    Parameters:
    -----------
    mean : torch.tensor, shape [batch_size, D]
        The mean of the distribution.
    logsigma : torch.tensor, shape [batch_size, D]
        The logarithm of the variance of the distribution.
    
    Returns:
    --------
    sample : torch.tensor, shape [batch_size, D]
        Samples from this distribution, differentiable w.r.t. to mean and log_var.
    """
    sigma = torch.exp(logsigma)
    eps = torch.zeros_like(sigma, device=mean.device, requires_grad=False).normal_()
    return mean + eps * sigma

def kl_divergence_diagonal_normal(mu: Tensor, logsigma: Tensor, prior_mu: Union[Tensor, float], prior_sigma: Union[Tensor, float]):
        """
        Computes the KL divergence between one diagonal Gaussian posterior
        and the diagonal Gaussian prior.

        Parameters:
        -----------
        mu : Tensor, shape [D]
            Mean of the posterior Gaussian.
        logsigma : Tensor, shape [D]
            Variance of the posterior Gaussian.
        prior_mu : Tensor, shape [D] or float
            Mean of the prior Gaussian.
        prior_sigma : Tensor, shape [D] or float
            Variance of the prior Gaussian.

        Returns:
        --------
        kl_divergence : Tensor, shape [1]
            The KL divergence between the posterior and the prior.
        """
        # See: https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/

        return 0.5 * torch.sum(
            logsigma.exp() / prior_sigma + # tr(prior_sigma^-1 * sigma0) as both are diagon
            ((mu - prior_mu).pow(2) / (prior_sigma)**2) - # quadratic form, as sigma1 is diagonal we can express it that way
            1 + # d
            torch.log(prior_sigma) - logsigma  # ln(|sigma1| / |sigma0|) = ln(|sigma1|) - ln(|sigma0|)
        )
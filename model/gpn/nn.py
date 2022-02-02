""" Implementation taken from https://github.com/stadlmax/Graph-Posterior-Network """


import torch
import torch.distributions as D
from .distribution import Dirichlet as ApproximateDirichlet

def uce_loss(
        alpha: torch.Tensor,
        y: torch.Tensor,) -> torch.Tensor:
    """utility function computing uncertainty cross entropy /
    bayesian risk of cross entropy

    Args:
        alpha (torch.Tensor): parameters of Dirichlet distribution
        y (torch.Tensor): ground-truth class labels (not one-hot encoded).

    Returns:
        torch.Tensor: loss
    """

    if alpha.dim() == 1:
        alpha = alpha.view(1, -1)

    a_sum = alpha.sum(-1)
    a_true = alpha.gather(-1, y.view(-1, 1)).squeeze(-1)
    uce = a_sum.digamma() - a_true.digamma() 
    return uce.mean()

def entropy_reg(
        alpha: torch.Tensor,
        beta_reg: float,
        approximate: bool = False,
        reduction: str = 'sum') -> torch.Tensor:
    """calculates entopy regularizer

    Args:
        alpha (torch.Tensor): dirichlet-alpha scores
        beta_reg (float): regularization factor
        approximate (bool, optional): flag specifying if the entropy is approximated or not. Defaults to False.

    Returns:
        torch.Tensor: REG
    """

    if approximate:
        reg = ApproximateDirichlet(alpha).entropy()
    else:
        reg = D.Dirichlet(alpha).entropy()

    reg = reg.mean()

    return -beta_reg * reg

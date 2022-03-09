import torch

def orthonormal_regularization_loss(weight: torch.Tensor, spectrum: float=1.0) -> float:
    """ Computes an orthonormal regularizer |W^T @ W - Id| 
    
    Parameters:
    -----------
    weight : torch.Tensor, shape [d1, d2]
        The weight matrix.
    spectrum : float, optional, default: 1.0
        The desired spectrum of the orthogonal matrix.
    
    Returns:
    --------
    ortho_loss : float
        The orthonormal regularization.
    """
    return torch.linalg.norm((weight.T @ weight) - spectrum * torch.eye(weight.size(1), device=weight.device), ord='fro')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Code taken from: https://github.com/AlexanderMath/fasth/blob/master/fasthpp.py

def fasthpp(V, X, stop_recursion=3): 
  """
    Computes the product U(V) @ X, where V are the householder weights of shape (d, d)

    V: matrix that represent weights of householder matrices (d, d)
    X: rectangular matrix (d, bs) to compute H(V) @ X
    stop_recursion: integer that controls how many merge iterations before recursion stops. 
    		    if None recursion continues until base case. 
  """
  d = V.shape[0]

  Y_ = V.clone().T
  W_ = -2*Y_.clone()

  # Only works for powers of two. 
  assert (d & (d-1)) == 0 and d != 0, "d should be power of two. You can just pad the matrix. " 

  # Step 1: compute (Y, W)s by merging! 
  k = 1
  for i, c in enumerate(range(int(np.log2(d)))):  
    k_2 = k 
    k  *= 2

    m1_ = Y_.view(d//k_2, k_2, d)[0::2] @ torch.transpose(W_.view(d//k_2, k_2, d)[1::2], 1, 2)
    m2_ = torch.transpose(W_.view(d//k_2, k_2, d)[0::2], 1, 2) @ m1_

    W_ = W_.view(d//k_2, k_2, d).clone()
    W_[1::2] += torch.transpose(m2_, 1, 2)
    W_ = W_.view(d, d)

    if stop_recursion is not None and c == stop_recursion: break

  # Step 2: 
  if stop_recursion is None:   return X + W_.T @ (Y_ @ X) 
  else: 
    # For each (W,Y) pair multiply with 
    for i in range(d // k-1, -1, -1 ):
      X = X + W_[i*k: (i+1)*k].T @ (Y_[i*k: (i+1)*k]  @ X )
    return X 

def biggest_pow2(x: int) -> int:
    """ Finds the smallest 2^k such that x<=2^k"""
    k = 0
    y = 1
    while y < x:
        k +=1
        y *=2
    return y

class OrthogonalLinear(nn.Module):
    """ Module that represents a linear transformation decomposed into U @ Sigma @ V, where
        Sigma = Id * singular values
    
    Parameters:
    -----------
    in_dim : int
        The input dimension
    out_dim : int
        The output dimension
    singular_values : float, optional, default: 1.0
        The singular values the transformation is associated with
    recursion_depth : int or None, optional, default: 3
        The recursion depth to calculate householder transformations.
    """

    def __init__(self, in_dim: int, out_dim: int, singular_values: float=1.0, recursion_depth: int=3):

        super().__init__()

        # Fasth algorithm only operates on powers of 2
        padded_in_dim = biggest_pow2(in_dim)
        self.padding_input = (0, padded_in_dim - in_dim)
        self.padding_V_in = (0, padded_in_dim - in_dim, 0, padded_in_dim - in_dim)
        padded_out_dim = biggest_pow2(out_dim)
        self.padding_V_out = (0, padded_out_dim - out_dim, 0, padded_out_dim - out_dim)
        self.out_dim = out_dim

        self.latent_trunc_dim = padded_out_dim
        self.recursion_depth = recursion_depth
        self.singular_values = singular_values
        
        # Householder parameters of shape (d, d)
        self.V_in = torch.nn.Parameter(torch.zeros((in_dim, in_dim)).normal_(0, 0.05))
        self.V_out = torch.nn.Parameter(torch.zeros((out_dim, out_dim)).normal_(0, 0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Parameters:
        -----------
        x : torch.Tensor, shape [B, in_dim]
            The input.
        
        Returns:
        --------
        x' : torch.Tensor, shape [B, out_dim]
            The output.
        """
        x = F.pad(x, self.padding_input, 'constant', 0)
        V_in_padded = F.pad(self.V_in, self.padding_V_in, 'constant', 0)
        V_out_padded = F.pad(self.V_out, self.padding_V_out, 'constant', 0)

        latent = fasthpp(V_in_padded, x.T, stop_recursion=self.recursion_depth) # (in_dim, B)
        latent_trunc = latent[: self.latent_trunc_dim, :]
        latent_trunc *= self.singular_values
        out = fasthpp(V_out_padded, latent_trunc, stop_recursion=self.recursion_depth)
        return out.T[:, :self.out_dim]


# # Testing
# if __name__ == '__main__':
#     lin = OrthogonalLinear(9120, 66)
#     x = torch.randn(92, 9120)
#     print(lin(x).size())
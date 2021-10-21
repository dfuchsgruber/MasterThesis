import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov
    
    From: https://github.com/pytorch/pytorch/issues/19037#issue-430654869
    """
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

class GMMFeatureSpaceDensity(torch.nn.Module):
    """ Model that estimates features space density fitting a Gaussian per class. """

    def __init__(self):
        super().__init__()
        self._fitted = False

    @torch.no_grad()
    def fit(self, features, labels):
        """ Fits the density models to a set of features and labels. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        labels : torch.Tensor, shape [N]
            Class labels.
        """
        if self._fitted:
            raise RuntimeError(f'GMM density model was already fitted.')
        self.components = {}
        self.coefs = nn.ParameterDict()
        self.means = nn.ParameterDict()
        self.trils = nn.ParameterDict()
        for label in torch.unique(labels):
            name = f'class_{label.item()}'
            # Save location and lower triangular covariance matrix to allow this module to have its state as state dict
            self.coefs[name] = nn.Parameter((labels == label).sum() / labels.size()[0], requires_grad=False)
            self.means[name] = nn.Parameter(features[labels == label].mean(dim=0), requires_grad=False)
            self.trils[name] = nn.Parameter(torch.linalg.cholesky(cov(features[labels == label])), requires_grad=False)
        self._fitted = True
    
    def forward(self, features):
        """ Gets the density at all feature points.
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features to get the density of.
        
        Returns:
        --------
        density : torch.Tensor, shape [N]
            Density of the GMM at each feature point.
        """
        # print(features.device)
        # for name in self.coefs:
        #     print(self.coefs[name].device, self.means[name].device, self.trils[name].device)
        if not self._fitted:
            raise RuntimeError(f'GMM density was not fitted to any data!')
        density = torch.zeros(features.size(0))
        for label in self.coefs:
            density += self.coefs[label] * torch.exp(MultivariateNormal(self.means[label], scale_tril=self.trils[label]).log_prob(features))
        return density

if __name__ == '__main__':
    x = torch.tensor([
        [1.2, 3.3],
        [0.8, 1.5],
        [-50, -51],
        [-49, -50],
        [-49, -51],
        [1.1, 0.8],
        [1.0, 1.0],
        [1.1, 1.1],
    ])
    y = torch.tensor([
        0, 0, 1, 1, 1, 0, 2, 2
    ], dtype=torch.int64)
    density = GMMFeatureSpaceDensity()
    density.fit(x, y)
    print(density(x))   

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy.stats
from warnings import warn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from model.dimensionality_reduction import DimensionalityReduction
import traceback

def cov_and_mean(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov and also returns the weighted mean.

    Parameters:
    -----------
    x : torch.Tensor, shape [N, D]
        The tensor to estimate covariance of.
    rowvar : bool
        If given, columns are treated as observations.
    bias : bool
        If given, use correction for the empirical covariance.
    aweights : torch.Tensor, shape [N] or None
        Weights for each observation.
    
    Returns:
    --------
    cov : torch.Tensor, shape [D, D]
        Empirical covariance.
    mean : torch.Tensor, shape [D]
        Empirical means.

    References:
    -----------
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

    return c.squeeze(), avg

class FeatureSpaceDensityGaussianPerClass(torch.nn.Module):
    """ Model that estimates features space density fitting a Gaussian per class. """

    name = 'GaussianPerClass'

    def __init__(
            self, 
            dimensionality_reduction={'type' : 'pca', 'number_components' : 2, 'per_class' : False}, 
            diagonal_covariance=False, 
            **kwargs
        ):
        super().__init__()
        self._fitted = False
        self.dimensionality_reduction = DimensionalityReduction(**dimensionality_reduction)
        self.diagonal_covariance = diagonal_covariance

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Dimensionality Reduction : ',
            str(self.dimensionality_reduction),
            f'\t Diagonal covariance : {self.diagonal_covariance}'
        ])

    def _make_covariance_psd(self, cov, eps=1e-6):
        """ If a covariance matrix is not psd (numerical errors), a small value is added to its diagonal to make it psd.
        
        Parameters:
        -----------
        cov : nn.parameters.Parameter, shape [D, D]
            Covariance matrix.
        eps : float
            The small value to add to the diagonal. If it still is not psd after that, eps is multiplied by 10 and the process repeats.
        """
        # Make covariance psd in case of numerical inaccuracies
        eps = 1e-6
        while not (np.linalg.eigvals(cov.numpy()) >= 0).all():
            print(f'Covariance not positive semi-definite. Adding {eps} to the diagnoal.')
            cov += torch.eye(cov.numpy().shape[0]) * eps
            eps *= 10

    @torch.no_grad()
    def fit(self, features, labels):
        """ Fits the density models to a set of features and labels. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        labels : torch.Tensor, shape [N, num_labels]
            Soft class labels.
        """
        if self._fitted:
            raise RuntimeError(f'GMM density model was already fitted.')
        self.components = {}
        self.coefs = dict()
        self.means = dict()
        self.covs = dict()

        self.dimensionality_reduction.fit(features, labels)
        transformed_by_class = self.dimensionality_reduction.transform(features)

        for label in range(labels.size(1)):
            if labels[:, label].sum(0) == 0: # No observations of this label, might happen if it was excluded from training
                continue

            transformed = torch.tensor(transformed_by_class[label]).float()
            self.covs[label], self.means[label] = cov_and_mean(transformed, aweights=labels[:, label])
            self.coefs[label] = labels[:, label].sum(0) / labels.size(0)

            if self.diagonal_covariance:
                self.covs[label] *= torch.eye(transformed.size(1))

            self._make_covariance_psd(self.covs[label], eps=1e-8)
            
        self._fitted = True
    
    @torch.no_grad()
    def forward(self, features):
        """ Gets the density at all feature points.
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features to get the density of.
        
        Returns:
        --------
        log_density : torch.Tensor, shape [N]
            Log density of the GMM at each feature point.
        """
        if not self._fitted:
            raise RuntimeError(f'GMM density was not fitted to any data!')

        transformed_by_class = self.dimensionality_reduction.transform(features)
        
        log_densities = [] # Per component, weighted with component
        for label in self.coefs:
            transformed = torch.tensor(transformed_by_class[label]).float()
            log_densities.append(MultivariateNormal(self.means[label], covariance_matrix=self.covs[label]).log_prob(transformed))
        log_densities = torch.stack(log_densities) # shape n_classes, N
        return torch.logsumexp(log_densities, 0) # Shape N


class FeatureSpaceDensityMixtureOfGaussians(torch.nn.Module):
    """ Model that estimates features space density fitting a Gaussian per class. """

    name = 'GaussianMixture'

    def __init__(
            self, 
            number_components=5, 
            seed=None,
            dimensionality_reduction={'type' : None, 'number_components' : 2, 'per_class' : False},
            **kwargs
        ):
        super().__init__()
        self._fitted = False
        self.number_components = number_components
        self.seed = seed
        # Dimensionality reduction: If used, assert that it is not per class
        if dimensionality_reduction['type'] is not None and dimensionality_reduction['per_class']:
            raise RuntimeError(f'GaussianMixture only supports a non-per-class dimensionality reduction.')
        self.dimensionality_reduction = DimensionalityReduction(**dimensionality_reduction)
        

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Number of components : {self.number_components}',
            f'\t Seed : {self.seed}',
        ])

    def _transform(self, features):
        transformed_by_class = self.dimensionality_reduction.transform(features)
        # All transformations should be the same, select any
        return list(transformed_by_class.values())[0]


    @torch.no_grad()
    def fit(self, features, labels):
        """ Fits the density models to a set of features and labels. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        labels : torch.Tensor, shape [N, num_classes]
            Soft class labels. Unused for this approach.
        """

        if self._fitted:
            raise RuntimeError(f'{self.name} density model was already fitted.')
        self.dimensionality_reduction.fit(features, labels)
        transformed = self._transform(features)
        self.gmm = GaussianMixture(n_components = self.number_components, random_state=self.seed)
        self.gmm.fit(transformed)
        self._fitted = True
    
    @torch.no_grad()
    def forward(self, features):
        """ Gets the density at all feature points.
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features to get the density of.
        
        Returns:
        --------
        log_density : torch.Tensor, shape [N]
            Log-density of the GMM at each feature point.
        """
        if not self._fitted:
            raise RuntimeError(f'{self.name} density was not fitted to any data!')
        transformed = self._transform(features)
        log_likelihood = torch.tensor(self.gmm.score_samples(transformed))
        return log_likelihood

densities = [
    FeatureSpaceDensityGaussianPerClass,
    FeatureSpaceDensityMixtureOfGaussians,
]

def get_density_model(density_type='unspecified', **kwargs):
    """ Gets a density model. Keyword arguments are passed to the respective density model.
    
    Parameters:
    -----------
    density_type : str
        The name of the density model.
    
    Returns:
    --------
    density_model : torch.nn.Module
        The density model.
    """
    for density_cls in densities:
        if density_cls.name.lower() == density_type.lower():
            return density_cls(**kwargs)
    else:
        raise RuntimeError(f'Unknown density model type {density_type.lower()}')

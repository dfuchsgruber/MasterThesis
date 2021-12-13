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
    """ Model that estimates features space density fitting a Gaussian per class. 
    
    Parameteres:
    ------------
    diagonal_covariance : bool
        If only a diagonal covariance is to be fitted.
    prior : float
        Value to add to the diagonal of the covariance of each gaussian.
    relative : bool
        If the relative log density log(p(x | c)) - log(p(x)) should be reported.
    mode : 'weighted' or 'max'
        Which density to report:
            - 'weighted' : Report log(sum_c p(x | c) * p(c))
            - 'max' : Report log(max_c p(x | c))
    """

    name = 'GaussianPerClass'

    def __init__(
            self, 
            diagonal_covariance = False,
            prior=0.0,
            relative = False,
            mode = 'weighted',
            **kwargs
        ):
        super().__init__()
        self._fitted = False
        self.diagonal_covariance = diagonal_covariance
        self.prior = prior
        self.relative = relative
        self.mode = mode.lower()

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Diagonal covariance : {self.diagonal_covariance}'
            f'\t Prior: {self.prior}',
        ])

    @property
    def compressed_name(self):
        tags = ['gpc']
        if self.diagonal_covariance:
            tags.append('diag')
        else:
            tags.append('full')
        if self.relative:
            tags.append('relative')
        if self.mode == 'weighted':
            tags.append('weighted')
        elif self.mode == 'max':
            tags.append('max')
        return '-'.join(tags)

    def _make_covariance_psd(self, cov, eps=1e-6, tol=1e-6):
        """ If a covariance matrix is not psd (numerical errors), a small value is added to its diagonal to make it psd.
        
        Parameters:
        -----------
        cov : nn.parameters.Parameter, shape [D, D]
            Covariance matrix.
        eps : float
            The small value to add to the diagonal. If it still is not psd after that, eps is multiplied by 10 and the process repeats.
        tol : float
            The minimal eigenvalue that is required for the covariance matrix.
        """
        # Make covariance psd in case of numerical inaccuracies
        eps = 1e-6
        while not (np.linalg.eigvals(cov.numpy()) > tol).all():
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

        features = features.cpu().float()
        
        self.components = {}
        self.coefs = dict()
        self.means = dict()
        self.covs = dict()

        for label in range(labels.size(1)):
            if labels[:, label].sum(0) == 0: # No observations of this label, might happen if it was excluded from training
                continue

            self.covs[label], self.means[label] = cov_and_mean(features, aweights=labels[:, label])
            self.covs[label] += torch.eye(features.size(1)) * self.prior
            self.coefs[label] = labels[:, label].sum(0) / labels.size(0)

            if self.diagonal_covariance:
                self.covs[label] *= torch.eye(features.size(1))

            self._make_covariance_psd(self.covs[label], eps=1e-8)
        
        if self.relative:
            self.covs['all'], self.means['all'] = cov_and_mean(features)
            self.covs['all'] += torch.eye(features.size(1)) * self.prior
            
            if self.diagonal_covariance:
                self.covs['all'] *= torch.eye(features.size(1))

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
            raise RuntimeError(f'GPC density was not fitted to any data!')
        
        features = features.cpu().float()

        log_densities, weights = [], []
        # Per component, weighted with component
        for label, weight in self.coefs.items():
            log_densities.append(
                MultivariateNormal(self.means[label], 
                covariance_matrix=self.covs[label], 
                validate_args=False).log_prob(
                    features.cpu()
                ))
            weights.append(weight)
        log_densities, weights = torch.stack(log_densities), torch.stack(weights).view((-1, 1))
        # `log_densities` has shape (num_classes, N)

        if self.relative: 
            # Subtract the class independet background density p(x)
            log_densities -= MultivariateNormal(
                self.means['all'],
                covariance_matrix=self.covs['all'], 
                validate_args=False
            ).log_prob(features.cpu())

        if self.mode == 'weighted':
            return torch.logsumexp(log_densities + torch.log(weights), 0) # Sum over `num_classes` axis
        elif self.mode == 'max':
            return torch.max(log_densities, 0)[0]
        else:
            raise RuntimeError(f'Unknown mode for {self.name} : {self.mode}')


class FeatureSpaceDensityMixtureOfGaussians(torch.nn.Module):
    """ Model that estimates features space density fitting a Gaussian per class. """

    name = 'GaussianMixture'

    def __init__(
            self, 
            number_components=5, 
            seed=1337,
            **kwargs
        ):
        super().__init__()
        self._fitted = False
        self.number_components = number_components
        self.seed = seed
        

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Number of components : {self.number_components}',
            f'\t Seed : {self.seed}',
        ])

    @property
    def compressed_name(self):
        return f'{self.number_components}-mog'

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
        self.gmm = GaussianMixture(n_components = self.number_components, random_state=self.seed)
        self.gmm.fit(features.cpu().numpy())
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
        log_likelihood = torch.tensor(self.gmm.score_samples(features.cpu().numpy()))
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

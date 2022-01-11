import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import TensorDataset
import scipy.stats
from warnings import warn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import traceback
import pyblaze.nn as xnn
from pyblaze.utils.stdlib import flatten
from model.normalizing_flow import NormalizingFlow

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


class FeatureSpaceDensityPerClass(torch.nn.Module):
    """ Base module to fit a density per class. """

    def __init__(
        self,
        seed : int = 1337,
        mode : str = 'weighted',
        relative : bool = False,
    ):
        super().__init__()
        self.seed = seed
        self.mode = mode.lower()
        self.relative = relative
        self._fitted = False
    
    @property
    def _tags(self):
        tags = []
        if self.relative:
            tags.append('relative')
        if self.mode == 'weighted':
            tags.append('weighted')
        elif self.mode == 'max':
            tags.append('max')
        return tags

    def get_class_weight(self, class_idx, soft):
        """ Fits the weight of a certain class density model. """ 
        return soft[:, class_idx].sum(0) / soft.size(0)

    def fit_class(self, class_idx, features, soft):
        """ Fits the density model for a given class. 
        
        Parameters:
        -----------
        class_idx : int or 'all'
            The class to fit for.
        features : torch.Tensor, shape [N, D]
            The features to fit.
        soft : torch.Tensor, shape [N, num_classes]
            Soft probabilities (scores) for assigning a sample to a class.
        """ 
        raise NotImplemented
    
    def get_density_class(self, class_idx, features):
        raise NotImplemented

    def fit(self, features, soft):
        """ Fits the density models to a set of features and labels. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        soft : torch.Tensor, shape [N, num_labels]
            Soft class labels.
        """
        if self._fitted:
            raise RuntimeError(f'Density model was already fitted.')

        self.class_weights = dict()
        for label in range(soft.size(1)):
            if soft[:, label].sum(0) == 0:
                # No observations
                continue
            self.class_weights[label] = self.get_class_weight(label, soft)
            self.fit_class(label, features, soft)

        if self.relative:
            self.fit_class('all', features, soft)

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
            raise RuntimeError(f'Density was not fitted to any data!')
        
        log_densities, weights = [], []
        for label, weight in self.class_weights.items():
            log_densities.append(self.get_density_class(label, features))
            weights.append(weight)
        log_densities, weights = torch.stack(log_densities), torch.stack(weights).view((-1, 1))

        if self.relative: 
            # Subtract the class independet background density p(x)
            log_densities -= self.get_density_class('all', features)
        
        if self.mode == 'weighted':
            return torch.logsumexp(log_densities + torch.log(weights), 0) # Sum over `num_classes` axis
        elif self.mode == 'max':
            return torch.max(log_densities, 0)[0]
        else:
            raise RuntimeError(f'Unknown mode for {self.name} : {self.mode}')

class FeatureSpaceDensityMixtureOfGaussiansPerClass(FeatureSpaceDensityPerClass):
    """ Model that estimates feature space density fitting a Mixture of Gaussians per class. 
    
    Parameters:
    -----------
    diagonal_covariance : bool
        If only a diagonal covariance is to be fitted.
    relative : bool
        If the relative log density log(p(x | c)) - log(p(x)) should be reported.
    mode : 'weighted' or 'max'
        Which density to report:
            - 'weighted' : Report log(sum_c p(x | c) * p(c))
            - 'max' : Report log(max_c p(x | c))
    """

    name = 'MixtureOfGaussiansPerClass'

    def __init__(
            self, 
            diagonal_covariance = False,
            relative = False,
            mode = 'weighted',
            seed = 1337,
            number_components = 2,
            **kwargs
        ):
        super().__init__()
        self._fitted = False
        self.diagonal_covariance = diagonal_covariance
        self.relative = relative
        self.mode = mode.lower()
        self.number_components = number_components
        self.seed = seed

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Diagonal covariance : {self.diagonal_covariance}',
            f'\t Relative : {self.relative}',
            f'\t Mode : {self.mode}',
            f'\t Number components : {self.number_components}',
        ])

    @property
    def compressed_name(self):
        tags = ['mogpc']
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
            raise RuntimeError(f'MoGPC density model was already fitted.')

        features = features.cpu().float()
        self.densities = {}
        self.coefs = dict()
        if self.diagonal_covariance:
            covariance_type = 'diag'
        else:
            self.diagonal_covariance = 'full'

        hard = labels.argmax(1)
        for label in range(labels.size(1)):
            if labels[:, label].sum(0) == 0: # No observations of this label, might happen if it was excluded from training
                continue
            features_label = features[hard == label]
            self.coefs[label] = (hard == label).sum(0) / labels.size(0)
            self.densities[label] = GaussianMixture(n_components = self.number_components, random_state=self.seed, covariance_type = covariance_type)
            self.densities[label].fit(features_label.numpy())

        if self.relative:
            self.densities['all'] = GaussianMixture(n_components = self.number_components, random_state=self.seed, covariance_type = covariance_type)
            self.densities['all'].fit(features.numpy())

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
        features = features.cpu()

        log_densities, weights = [], []
        # Per component, weighted with component
        for label, weight in self.coefs.items():
            log_densities.append(torch.tensor(self.densities[label].score_samples(features.numpy())))
            weights.append(weight)
        log_densities, weights = torch.stack(log_densities), torch.stack(weights).view((-1, 1))

        if self.relative:
            log_densities -= torch.tensor(self.densities['all'].score_samples(features.numpy()))
        
        if self.mode == 'weighted':
            return torch.logsumexp(log_densities + torch.log(weights), 0) # Sum over `num_classes` axis
        elif self.mode == 'max':
            return torch.max(log_densities, 0)[0]
        else:
            raise RuntimeError(f'Unknown mode for {self.name} : {self.mode}')

class FeatureSpaceDensityGaussianPerClass(FeatureSpaceDensityPerClass):
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
            **kwargs
        ):
        super().__init__(**kwargs)
        self.diagonal_covariance = diagonal_covariance
        self.prior = prior
        self.covs = dict()
        self.means = dict()

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
        tags += self._tags
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
    def fit_class(self, class_idx, features, soft):
        """ Fits the density model for a given class. 
        
        Parameters:
        -----------
        class_idx : int or 'all'
            The class to fit for.
        features : torch.Tensor, shape [N, D]
            The features to fit.
        soft : torch.Tensor, shape [N, num_classes]
            Soft probabilities (scores) for assigning a sample to a class.
        """ 
        if class_idx == 'all':
            self.covs[class_idx], self.means[class_idx] = cov_and_mean(features)
        else:
            self.covs[class_idx], self.means[class_idx] = cov_and_mean(features, aweights=soft[:, class_idx])
        self.covs[class_idx] += torch.eye(features.size(1)) * self.prior
        if self.diagonal_covariance:
            self.covs[class_idx] *= torch.eye(features.size(1))
        self._make_covariance_psd(self.covs[class_idx], eps=1e-8)

    def get_density_class(self, class_idx, features):
        """ Gets the density for a given class.
        
        Parameters:
        -----------
        class_idx : int or 'all'
            The class to get density for.
        features : torch.Tensor, shape [N, D]
            The features to evaluate density for.
        
        Returns:
        --------
        log_densities : torch.Tensor, shape [N]
            Log densities for each sample given that class density.
        """
        return MultivariateNormal(self.means[class_idx], 
                covariance_matrix=self.covs[class_idx], 
                validate_args=False).log_prob(
                    features.cpu()
                )

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

class FeatureSpaceDensityNormalizingFlowPerClass(FeatureSpaceDensityPerClass):

    name = 'FeatureSpaceDensityNormalizingFlowPerClass'

    def __init__(self, flow_type='maf', num_layers=2, hidden_dim=64, num_hidden=2, iterations=1000, seed=1337, gpu=False, verbose=False, *args, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.flow_type = flow_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.iterations = iterations
        self.gpu = gpu
        self.verbose = verbose

        self._fitted = False
        self.flows = dict()
        self.coefs = dict()

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Flow type : {self.flow_type}',
            f'\t Number of layers : {self.num_layers}',
            f'\t Hidden dimensionality : {self.hidden_dim}',
            f'\t Number of hidden units per layer : {self.num_hidden} '
            f'\t Iterations : {self.iterations}',
            f'\t On GPU : {self.gpu}'
        ])

    @property
    def compressed_name(self):
        tags = ['nfpc', str(self.num_layers), str(self.flow_type)]
        tags += self._tags
        return '-'.join(tags)


    def fit_class(self, class_idx, features, soft):
        """ Fits the density model for a given class. 
        
        Parameters:
        -----------
        class_idx : int or 'all'
            The class to fit for.
        features : torch.Tensor, shape [N, D]
            The features to fit.
        soft : torch.Tensor, shape [N, num_classes]
            Soft probabilities (scores) for assigning a sample to a class.
        """
        hard = soft.argmax(1)
        if class_idx != 'all':
            self.coefs[class_idx] = (hard == class_idx).sum(0) / hard.size(0)
        
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.flows[class_idx] = NormalizingFlow(self.flow_type, self.num_layers, features.size(1), seed=self.seed, num_hidden=self.num_hidden, 
                hidden_dim = self.hidden_dim, gpu = self.gpu)
        if class_idx == 'all':
            weights = torch.ones(features.size(0)).float()
        else:
            weights = soft[:, class_idx]
        self.flows[class_idx].fit(features, weights=weights, verbose=self.verbose, iterations=self.iterations)

    def get_density_class(self, class_idx, features):
        """ Gets the density for a given class.
        
        Parameters:
        -----------
        class_idx : int or 'all'
            The class to get density for.
        features : torch.Tensor, shape [N, D]
            The features to evaluate density for.
        
        Returns:
        --------
        log_densities : torch.Tensor, shape [N]
            Log densities for each sample given that class density.
        """
        return self.flows[class_idx](features).cpu()


densities = [
    FeatureSpaceDensityGaussianPerClass,
    FeatureSpaceDensityMixtureOfGaussians,
    FeatureSpaceDensityNormalizingFlowPerClass,
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

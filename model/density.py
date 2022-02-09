from typing import Optional
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
from itertools import product

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

def _make_covariance_psd_symmetric(cov, eps=1e-12, tol=1e-6):
    """ If a covariance matrix is not psd (numerical errors) and symmetric, a small value is added to its diagonal to make it psd.

    Parameters:
    -----------
    cov : nn.parameters.Parameter, shape [D, D]
        Covariance matrix.
    eps : float
        The small value to add to the diagonal. If it still is not psd after that, eps is multiplied by 10 and the process repeats.
    tol : float
        The minimal eigenvalue that is required for the covariance matrix.
        
    Returns:
    --------
    cov : torch.Tensor, shape [D, D]
        Symmetric positive definite covariance.
    """
    # Make covariance psd in case of numerical inaccuracies
    while not (np.allclose(cov.numpy(), cov.numpy().T) and
              np.all(np.linalg.eigvalsh(cov.numpy()) > tol)):
        # print(f'Matrix not positive semi-definite. Adding {eps} to the diagnoal.')
        cov += torch.eye(cov.numpy().shape[0]) * eps
        cov = 0.5 * (cov + cov.T) # Hacky way to make the matrix symmetric without changing its values too much (the diagonal stays intact for sure)
        eps *= 10
    return cov

class FeatureSpaceDensity(torch.nn.Module):
    """ Base class to fit feature space densities. """

    def __init__(self, evaluation_kwargs_grid={}):
        super().__init__()
        self.evaluation_kwargs_grid = evaluation_kwargs_grid

    @property
    def evaluation_kwargs(self):
        result = []
        keys = list(self.evaluation_kwargs_grid.keys())
        for values in product(*[self.evaluation_kwargs_grid[k] for k in keys]):
            kwargs = {key : values[idx] for idx, key in enumerate(keys)}
            result.append(('-'.join([''] + [f'{kwarg}:{v}' for kwarg, v in kwargs.items()]), kwargs))
        if len(result) == 0:
            result.append(('', {})) # No specific evaluation modes with differing kwargs to `forward`
        return result

class FeatureSpaceDensityPerClass(FeatureSpaceDensity):
    """ Base module to fit a density per class. """

    def __init__(
        self,
        seed : int = 1337,
        evaluation_kwargs_grid = {
            'mode' : ['max', 'weighted'],
            'relative' : [True, False],
        },
        **kwargs,
    ):
        super().__init__(evaluation_kwargs_grid = evaluation_kwargs_grid, **kwargs)
        self.seed = seed
        self._fitted = False
    
    @property
    def _tags(self):
        tags = []
        return tags

    def get_class_weight(self, class_idx, soft, soft_val):
        """ Fits the weight of a certain class density model. """ 
        return (soft[:, class_idx].sum(0) + soft_val[:, class_idx].sum(0)) / (soft.size(0) + soft_val.size(0))

    def fit_class(self, class_idx: int, features: torch.Tensor, soft: torch.Tensor, features_val: torch.Tensor, soft_val: torch.Tensor):
        """ Fits the density model for a given class. 
        
        Parameters:
        -----------
        class_idx : int or 'all'
            The class to fit for.
        features : torch.Tensor, shape [N, D]
            The features to fit.
        soft : torch.Tensor, shape [N, num_classes]
            Soft probabilities (scores) for assigning a sample to a class.
        features_val : torch.Tensor, shape [N, D']
            The features for validation.
        soft_val : torch.Tensor, shape [N', num_classes]
            Soft probabilities (scores) for assigning a validation sample to a class.
        """ 
        raise NotImplemented
    
    def get_density_class(self, class_idx, features):
        raise NotImplemented

    def fit(self, features: torch.Tensor, soft: torch.Tensor, features_val: torch.Tensor, soft_val: torch.Tensor):
        """ Fits the density models to a set of features and labels. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        soft : torch.Tensor, shape [N, num_labels]
            Soft class labels.
        features_val : torch.Tensor, shape [N, D']
            The features for validation.
        soft_val : torch.Tensor, shape [N', num_classes]
            Soft probabilities (scores) for assigning a validation sample to a class.
        """
        if self._fitted:
            raise RuntimeError(f'Density model was already fitted.')

        self.class_weights = dict()
        for label in range(soft.size(1)):
            if soft[:, label].sum(0) == 0:
                # No observations
                continue
            self.class_weights[label] = self.get_class_weight(label, soft, soft_val)
            self.fit_class(label, features, soft, features_val, soft_val)

        self.fit_class('all', features, soft, features_val, soft_val)
        self._fitted = True

    @torch.no_grad()
    def forward(self, features, mode='weighted', relative=False):
        """ Gets the density at all feature points.
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features to get the density of.
        mode : 'weighted' or 'max'
            If the per class densities are aggregated by weighting or by taking the max.
        relative : bool
            If given, the aggregated per class density is weighted against a background density fit to all classes.
        
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

        if relative: 
            # Subtract the class independet background density p(x)
            log_densities -= self.get_density_class('all', features)
        
        if mode == 'weighted':
            return torch.logsumexp(log_densities + torch.log(weights), 0) # Sum over `num_classes` axis
        elif mode == 'max':
            return torch.max(log_densities, 0)[0]
        else:
            raise RuntimeError(f'Unknown mode for {self.name} : {mode}')

class FeatureSpaceDensityGaussianPerClass(FeatureSpaceDensityPerClass):
    """ Model that estimates features space density fitting a Gaussian per class. 
    
    Parameteres:
    ------------
    covariance : str
        Restrictions on the covariance.
    regularization : float
        Value to add to the diagonal of the covariance of each gaussian.
    """

    name = 'GaussianPerClass'

    def __init__(
            self, 
            covariance='diag',
            regularization = False,
            fit_val = True,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.covariance = covariance
        self.regularization = regularization
        self.fit_val = fit_val
        self.covs = dict()
        self.means = dict()

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\tCovariance type : {self.diagonal_covariance}'
            f'\tRegularization: {self.regularization}',
            f'\tFit validation data: {self.fit_val}',
        ])

    @property
    def compressed_name(self):
        tags = ['gpc', str(self.covariance)]
        tags += self._tags
        return '-'.join(tags)

    @torch.no_grad()
    def fit_class(self, class_idx: int, features: torch.Tensor, soft: torch.Tensor, features_val: torch.Tensor, soft_val: torch.Tensor):
        """ Fits the density model for a given class. 
        
        Parameters:
        -----------
        class_idx : int or 'all'
            The class to fit for.
        features : torch.Tensor, shape [N, D]
            The features to fit.
        soft : torch.Tensor, shape [N, num_classes]
            Soft probabilities (scores) for assigning a sample to a class.
        features_val : torch.Tensor, shape [N, D']
            The features for validation.
        soft_val : torch.Tensor, shape [N', num_classes]
            Soft probabilities (scores) for assigning a validation sample to a class.
        """ 
        
        if self.fit_val: # Gaussian per Class fits the validation data as well
            features = torch.cat((features, features_val), 0)
            soft = torch.cat((soft, soft_val), 0)

        if class_idx == 'all':
            self.covs[class_idx], self.means[class_idx] = cov_and_mean(features, aweights=soft.sum(1))
        else:
            self.covs[class_idx], self.means[class_idx] = cov_and_mean(features, aweights=soft[:, class_idx])
        self.covs[class_idx] += torch.eye(features.size(1)) * self.regularization
        if self.covariance.lower() in ('diag', 'diagonal'):
            self.covs[class_idx] *= torch.eye(features.size(1))
        elif self.covariance.lower() in ('eye', 'identity', 'id'):
            self.covs[class_idx] = torch.eye(features.size(1))
        elif self.covariance.lower() in ('full'):
            pass
        elif self.covariance.lower() in ('iso'):
            scale = torch.diag(self.covs[class_idx]).mean()
            self.covs[class_idx] = scale * torch.eye(features.size(1))
        else:
            raise ValueError(f'Unsupported covariance type {self.covariance}')
        self.covs[class_idx] = _make_covariance_psd_symmetric(self.covs[class_idx], eps=1e-6)

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

class FeatureSpaceDensityMixtureOfGaussians(FeatureSpaceDensity):
    """ Model that estimates features space density fitting a Gaussian per class. """

    name = 'GaussianMixture'

    def __init__(
            self, 
            number_components=-1, 
            seed=1337,
            diagonal_covariance = False,
            initialization = 'random',
            fit_val = True,
            **kwargs
        ):
        super().__init__(**kwargs)
        self._fitted = False
        self.number_components = number_components
        self.fit_val = fit_val
        self.seed = seed
        self.diagonal_covariance = diagonal_covariance
        self.initialization = initialization
        
    def __str__(self):
        return '\n'.join([
            self.name,
            f'\tNumber of components : {self.number_components}',
            f'\tSeed : {self.seed}',
            f'\tDiagonal Covariance : {self.diagonal_covariance}',
            f'\tInitialization method : {self.initialization}',
            f'\tFit validation data : {self.fit_val}',
        ])

    @property
    def compressed_name(self):
        tags = ['mog', str(self.number_components)]
        if self.diagonal_covariance:
            tags.append('diag')
        else:
            tags.append('full')
        tags.append(self.initialization)
        return '-'.join(tags)

    @torch.no_grad()
    def fit(self, features: torch.Tensor, soft: torch.Tensor, features_val: torch.Tensor, soft_val: torch.Tensor):
        """ Fits the density models to a set of features and labels. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        soft : torch.Tensor, shape [N, num_labels]
            Soft class labels.
        features_val : torch.Tensor, shape [N, D']
            The features for validation.
        soft_val : torch.Tensor, shape [N', num_classes]
            Soft probabilities (scores) for assigning a validation sample to a class.
        """
        if self._fitted:
            raise RuntimeError(f'{self.name} density model was already fitted.')
            
        if self.fit_val:
            features = torch.cat((features, features_val), 0)
            soft = torch.cat((soft, soft_val), 0)

        if self.number_components <= 0:
            self.number_components = soft.size(1)

        if self.diagonal_covariance:
            covariance_type = 'diag'
        else:
            covariance_type = 'full'
        if self.initialization.lower() == 'random':
            weights = None
            means = None
            precisions = None
        elif self.initialization.lower() == 'predictions':
            if soft.size(1) != self.number_components:
                raise RuntimeError(f'Cant initialize GMM with {self.number_components} components by {soft.size(1)} dimensional scores')
                
            weights = soft.sum(0)
            weights /= weights.sum()
            precisions, means = [], []
            for class_idx in range(soft.size(1)):
                cov, mean = cov_and_mean(features, aweights=soft[:, class_idx])
                if self.diagonal_covariance:
                    cov *= torch.eye(features.size(1))
                cov = _make_covariance_psd_symmetric(cov, eps=1e-6)
                precision = torch.inverse(cov)
                means.append(mean)
                precision = _make_covariance_psd_symmetric(precision, eps=1e-6)
                if self.diagonal_covariance:
                    precision = torch.diagonal(precision) # GMM only wants the diagonal as initialization
                precisions.append(precision)
                
            weights, means, precisions = weights.numpy().astype(float), torch.stack(means, 0).numpy(), torch.stack(precisions, 0).numpy()
            weights /= weights.sum() # This helps passing the weight normalization argument assertion in `GaussianMixture.fit`
        else:
            raise RuntimeError(f'Unsupported initialization method for GMM {self.initialization}') 

        self.gmm = GaussianMixture(n_components = self.number_components, random_state=self.seed, covariance_type = covariance_type, 
                                    weights_init = weights, means_init = means, precisions_init = precisions, verbose=0)
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

    name = 'NormalizingFlowPerClass'

    def __init__(self, flow_type='maf', num_layers=2, hidden_dim=None, num_hidden=2, max_iterations=1000, seed=1337, 
                    gpu=True, weight_decay=1e-3, verbose=False, patience=5, *args, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.flow_type = flow_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.max_iterations = max_iterations
        self.weight_decay = weight_decay
        self.gpu = gpu
        self.verbose = verbose
        self.patience = patience

        self._fitted = False
        self.flows = dict()
        self.coefs = dict()

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\tFlow type : {self.flow_type}',
            f'\tNumber of layers : {self.num_layers}',
            f'\tHidden dimensionality : {self.hidden_dim}',
            f'\tNumber of hidden units per layer : {self.num_hidden} ',
            f'\tMaximal number of itearations : {self.max_iterations}',
            f'\tWeight Decay : {self.weight_decay}',
            f'\tEarly stopping patience : {self.patience}',
            f'\t On GPU : {self.gpu}',
        ])

    @property
    def compressed_name(self):
        tags = ['nfpc', str(self.num_layers), str(self.flow_type), f'{self.num_hidden}', f'{self.hidden_dim}']
        tags += self._tags
        return '-'.join(tags)

    def fit_class(self, class_idx: int, features: torch.Tensor, soft: torch.Tensor, features_val: torch.Tensor, soft_val: torch.Tensor):
        """ Fits the density model for a given class. 
        
        Parameters:
        -----------
        class_idx : int or 'all'
            The class to fit for.
        features : torch.Tensor, shape [N, D]
            The features to fit.
        soft : torch.Tensor, shape [N, num_classes]
            Soft probabilities (scores) for assigning a sample to a class.
        features_val : torch.Tensor, shape [N, D']
            The features for validation.
        soft_val : torch.Tensor, shape [N', num_classes]
            Soft probabilities (scores) for assigning a validation sample to a class.
        """ 
        if class_idx != 'all':
            hard = soft.argmax(1)
            self.coefs[class_idx] = (hard == class_idx).sum(0) / hard.size(0)
        
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.flows[class_idx] = NormalizingFlow(self.flow_type, self.num_layers, features.size(1), seed=self.seed, num_hidden=self.num_hidden, 
                hidden_dim = self.hidden_dim, gpu = self.gpu, weight_decay = self.weight_decay)
        if class_idx == 'all':
            weights = soft.sum(1)
            weights_val = soft_val.sum(1)
        else:
            weights = soft[:, class_idx]
            weights_val = soft_val[:, class_idx]
        self.flows[class_idx].fit(features, weights=weights, x_val=features_val, weights_val=weights_val, verbose=self.verbose, max_iterations=self.max_iterations)

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

class FeatureSpaceDensityNormalizingFlow(FeatureSpaceDensity):
    """ Feature space Density that fits a normalizing flow to all samples. """

    name = 'NormalizingFlow'

    def __init__(self, flow_type='maf', num_layers=2, hidden_dim=64, num_hidden=2, max_iterations=1000, seed=1337, gpu=True, verbose=False, weight_decay=1e-3, *args, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.flow_type = flow_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.max_iterations = max_iterations
        self.gpu = gpu
        self.verbose = verbose
        self.weight_decay = weight_decay

        self._fitted = False

    
    def __str__(self):
        return '\n'.join([
            self.name,
            f'\tFlow type : {self.flow_type}',
            f'\tNumber of layers : {self.num_layers}',
            f'\tHidden dimensionality : {self.hidden_dim}',
            f'\tNumber of hidden units per layer : {self.num_hidden} ',
            f'\tIterations : {self.max_iterations}',
            f'\tWeight Decay : {self.weight_decay}',
            f'\t On GPU : {self.gpu}',
        ])

    @property
    def compressed_name(self):
        tags = ['nf', str(self.num_layers), str(self.flow_type), f'{self.num_hidden}', f'{self.hidden_dim}']
        return '-'.join(tags)

    @torch.no_grad()
    def fit(self, features: torch.Tensor, soft: torch.Tensor, features_val: torch.Tensor, soft_val: torch.Tensor):
        """ Fits the density models to a set of features and labels. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        soft : torch.Tensor, shape [N, num_labels]
            Soft class labels.
        features_val : torch.Tensor, shape [N, D']
            The features for validation.
        soft_val : torch.Tensor, shape [N', num_classes]
            Soft probabilities (scores) for assigning a validation sample to a class.
        """
        self.flow = NormalizingFlow(self.flow_type, self.num_layers, features.size(1), seed=self.seed, num_hidden=self.num_hidden, 
                hidden_dim = self.hidden_dim, gpu = self.gpu, weight_decay = self.weight_decay)
        self.flow.fit(features, weights=soft.sum(1), x_val=features_val, weights_val=soft_val.sum(1), verbose=self.verbose, max_iterations=self.max_iterations)
        self._fitted = True

    def forward(self, features):
        return self.flow(features).cpu()

densities = [
    FeatureSpaceDensityGaussianPerClass,
    FeatureSpaceDensityMixtureOfGaussians,
    FeatureSpaceDensityNormalizingFlowPerClass,
    FeatureSpaceDensityNormalizingFlow,
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

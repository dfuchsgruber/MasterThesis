
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from model.parametrization import spectral_norm
import model.constants as mconst
from configuration import ModelConfiguration
import logging
from model.bayesian import kl_divergence_diagonal_normal, sample_normal
import math
from typing import Optional
from model.orthogonal import OrthogonalLinear

class GCNConv(torch_geometric.nn.GCNConv):
    """ GCN convolution that allows to clear and disable the cache that was used during training. """

    can_sample = False

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        logging.info(f'{self.__class__} disabled cache.')
        self.cached = False
        self._cached_edge_index = None
        self._cached_adj_t

class OrthogonalGCNConv(GCNConv):
    """GCN convolution with an orthogonal linear layer that uses SVD. It forces all its singular values to be a certain value. """
    def __init__(self, in_channels: int, out_channels: int, singular_values: float=1.0, *args, bias: bool=False, **kwargs):
        super().__init__(in_channels, out_channels, args, bias=bias, **kwargs)
        assert hasattr(self, 'lin') # Will be overriden with the Bayesian Linear layer
        self.lin = OrthogonalLinear(in_channels, out_channels, singular_values=singular_values)

class APPNPConv(torch_geometric.nn.APPNP):
    """ APPNP convolution that allows to clear and disable the cache that was used during training. """

    can_sample = False

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        logging.info(f'{self.__class__} disabled cache.')
        self.cached = False
        self._cached_edge_index = None
        self._cached_adj_t


class BayesianLinear(nn.Module):
    """ Bayesian Linear layer. """

    can_sample = True

    def __init__(self, input_dim: int, output_dim: int, bias=True, 
            prior_mean_bias=0, prior_variance_bias=1.0, 
            prior_mean_weight=0.0, prior_variance_weight=1.0):
        super().__init__()
        self._bias = bias

        # Priors
        if not isinstance (prior_mean_bias, torch.Tensor):
            prior_mean_bias = torch.Tensor([prior_mean_bias])
        if not isinstance (prior_variance_bias, torch.Tensor):
            prior_variance_bias = torch.Tensor([prior_variance_bias])
        if not isinstance (prior_mean_weight, torch.Tensor):
            prior_mean_weight = torch.Tensor([prior_mean_weight])
        if not isinstance (prior_variance_weight, torch.Tensor):
            prior_variance_weight = torch.Tensor([prior_variance_weight])
        self.register_buffer('w_mu_prior', prior_mean_weight)
        self.register_buffer('w_sigma_prior', prior_variance_weight)
        if self._bias:
            self.register_buffer('b_mu_prior', prior_mean_bias)
            self.register_buffer('b_sigma_prior', prior_variance_bias)

        # Gaussian parameters
        self.w_mu = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.w_logsigma = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self._bias:
            self.b_mu = nn.Parameter(torch.Tensor(output_dim))
            self.b_logsigma = nn.Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):

        std = 1 / math.sqrt(self.w_mu.size(1))
        self.w_mu.data.uniform_(-std, std)
        self.w_logsigma.data.fill_(self.w_sigma_prior.item())

        if self._bias:
            self.b_mu.data.uniform_(-std, std)
            self.b_logsigma.data.fill_(self.b_sigma_prior.item())
    

    def forward(self, x: torch.Tensor, sample=False) -> torch.Tensor:
        if sample:
            w = sample_normal(self.w_mu, self.w_logsigma)
        else:
            w = self.w_mu
        out = torch.matmul(x, w)
        if self._bias:
            if sample:
                b = sample_normal(self.b_mu, self.b_logsigma)
            else:
                b = self.b_mu
            out += b
        return out

    def kl_loss(self) -> float:
        """ Calculates the KL divergence loss between the parameters and a prior Gaussian. """
        loss = kl_divergence_diagonal_normal(self.w_mu, self.w_logsigma, self.w_mu_prior, self.w_sigma_prior)
        if self._bias:
            loss += kl_divergence_diagonal_normal(self.b_mu, self.b_logsigma, self.b_mu_prior, self.b_sigma_prior)
        return loss

    def num_kl_terms(self) -> float:
        """ Calculates how many kl terms this module has. """
        # How many terms are summed over in the kl loss 
        num = len(self.w_mu.view(-1))
        if self._bias:
            num += len(self.b_mu.view(-1))
        return num

class BaysianGCNConv(GCNConv):
    """ GCN convolution with a Bayesian linear layer. """

    def __init__(self, in_channels: int, out_channels: int, *args, bias=True, 
        prior_mean_bias=0, prior_variance_bias=1.0, prior_mean_weight=0.0, prior_variance_weight=1.0,
    **kwargs):
        super().__init__(in_channels, out_channels, *args, bias=False, **kwargs)
        assert hasattr(self, 'lin') # Will be overriden with the Bayesian Linear layer
        self.lin = BayesianLinear(in_channels, out_channels, bias=bias,
            prior_mean_bias=prior_mean_bias, prior_variance_bias=prior_variance_bias, 
            prior_mean_weight=prior_mean_weight, prior_variance_weight=prior_variance_weight,)

    def kl_loss(self) -> float:
        """ Calculates the KL divergence loss between the parameters and a prior Gaussian. """
        return self.lin.kl_loss()

    def num_kl_terms(self) -> float:
        """ Calculates how many kl terms this module has. """
        return self.lin.num_kl_terms()

class LinearWithSpectralNormaliatzion(nn.Module):
    """ Wrapper for a linear layer that applies spectral normalization and rescaling to the weight. """

    can_sample = False
    
    def __init__(self, input_dim, output_dim, config: ModelConfiguration, use_spectral_norm: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=config.use_bias)
        self._cached = False # Unused, for the interface
        if use_spectral_norm:
            self.linear = spectral_norm(self.linear, name='weight', rescaling=config.weight_scale)

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        pass # Has no edge cache

    def forward(self, x, *args, **kwargs):
        return self.linear(x)

class BasicBlock(nn.Module):
    """ Wrapper for basic block for GNNs. 
    
    Parameters:
    -----------
    input_dim : int, unused
        Input dimensionality.
    output_dim : int, unused
        Output dimensionality.
    conv : nn.Module
        The convolution operator to apply.
    act : nn.Module
        The activation to apply.
    config : ModelConfiguration
        Configuration for the model.
    """
    def __init__(self, input_dim: int, output_dim: int, conv: nn.Module, act: nn.Module, config: ModelConfiguration):
        super().__init__()
        self.conv = conv
        self.act = act
        self.dropout = config.dropout

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        self.conv.clear_and_disable_cache()

    def forward_impl(self, *args, sample=None, **kwargs):
        if sample is None:
            sample = self.training
        if self.conv.can_sample:
            h = self.conv(*args, sample=sample, **kwargs)
        else:
            h = self.conv(*args, **kwargs)
        if self.act is not None:
            h = self.act(h)
        if self.dropout:
            h = F.dropout(h, p=self.dropout, inplace=False, training=sample)
        return h

    def forward(self, *args, **kwargs):
        return self.forward_impl(*args, **kwargs)

class ResidualBlock(BasicBlock):
    """ Wrapper for any convolution that implements a residual connection. 
    
    Parameters:
    -----------
    input_dim : int
        Input dimensionality of the residual block
    output_dim : int
        Output dimensionality of the residual block
    conv : nn.Module
        The convolution to apply in the block
    act : nn.Module, optional
        Activation function
    config : ModelConfiguration, optional
        The configuration.
    """
    def __init__(self, input_dim: int, output_dim: int, conv: nn.Module, act: Optional[nn.Module], config: ModelConfiguration, use_spectral_norm: bool=True):
        super().__init__(input_dim, output_dim, conv, act, config)
        self.conv = conv
        self.act = act
        if input_dim != output_dim:
            self.input_projection = LinearWithSpectralNormaliatzion(input_dim, output_dim, config, use_spectral_norm=use_spectral_norm)

            if config.freeze_residual_projection:
                for param in self.input_projection.parameters():
                    param.requires_grad = False
        else:
            self.input_projection = None

    def forward(self, x, *args, **kwargs):
        # Use the basic block
        h = self.forward_impl(x, *args, **kwargs)
        if self.input_projection:
            x = self.input_projection(x)
        return x + h

def make_convolutions(input_dim: int, num_classes: int, cfg: ModelConfiguration, make_conv, *args, **kwargs) -> nn.ModuleList:
    """ Makes convolutions from a class and a set of input, hidden and output dimensions. """
    all_dims = [input_dim] + list(cfg.hidden_sizes) + [num_classes]
    convs = []
    dims = list(zip(all_dims[:-1], all_dims[1:]))
    for idx, (in_dim, out_dim) in enumerate(dims):

        # Spectral norm
        if idx == len(dims) - 1 and not cfg.use_spectral_norm_on_last_layer:
            sn = False
        else:
            sn = cfg.use_spectral_norm

        # Activation
        if idx == len(dims) - 1:
            act = None
        else:
            act = make_activation_by_configuration(cfg)

        conv = make_conv(in_dim, out_dim, cfg, use_spectral_norm=sn, *args, **kwargs)
        if cfg.residual and (idx < len(dims) - 1 or cfg.use_residual_on_last_layer):
            conv = ResidualBlock(in_dim, out_dim, conv, act, cfg, use_spectral_norm=sn)
        else:
            conv = BasicBlock(in_dim, out_dim, conv, act, cfg)
        convs.append(conv)
    return nn.ModuleList(convs)

def make_activation_by_configuration(configuration: ModelConfiguration) -> nn.Module:
    """ Makes the activation function form a configuration dict. """
    if configuration.activation == mconst.LEAKY_RELU:
        return nn.LeakyReLU()
    elif configuration.activation == mconst.RELU:
        return nn.ReLU()
    else:
        raise ValueError(f'Unsupported activation function {configuration.activation}')
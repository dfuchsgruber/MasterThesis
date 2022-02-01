import torch.nn as nn
import torch
import torch_geometric
from model.spectral_norm import spectral_norm
from configuration import ModelConfiguration
import logging

class GCNConv(torch_geometric.nn.GCNConv):
    """ GCN convolution that allows to clear and disable the cache that was used during training. """

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        logging.info('GCN conv disabled cache.')
        self.cached = False
        self._cached_edge_index = None
        self._cached_adj_t

class APPNPConv(torch_geometric.nn.APPNP):
    """ APPNP convolution that allows to clear and disable the cache that was used during training. """

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        logging.info('APPNP conv disabled cache.')
        self.cached = False
        self._cached_edge_index = None
        self._cached_adj_t

class ResidualBlock(nn.Module):
    """ Wrapper for any convolution that implements a residual connection. 
    
    Parameters:
    -----------
    input_dim : int
        Input dimensionality of the residual block
    output_dim : int
        Output dimensionality of the residual block
    conv : nn.Module
        The convolution to apply in the block
    use_spectral_norm : bool
        If `True`, spectral norm is applied to a potential input projection.
    weight_scale : float
        Bound on the spectral norm of input projection layer.
    use_bias : bool
        If a bias should be used for the input projection layer.
    """

    def __init__(self, input_dim: int, output_dim: int, conv: nn.Module, config: ModelConfiguration):
        super().__init__()
        self.conv = conv
        if input_dim != output_dim:
            self.input_projection = LinearWithSpectralNormaliatzion(input_dim, output_dim, config)

            if config.freeze_residual_projection:
                for param in self.input_projection.parameters():
                    param.requires_grad = False
        else:
            self.input_projection = None
    
    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        self.conv.clear_and_disable_cache()

    def forward(self, x, *args, **kwargs):
        h = self.conv(x, *args, **kwargs)
        if self.input_projection:
            x = self.input_projection(x)
        return x + h

class LinearWithSpectralNormaliatzion(nn.Module):
    """ Wrapper for a linear layer that applies spectral normalization and rescaling to the weight. """

    def __init__(self, input_dim, output_dim, config: ModelConfiguration):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=config.use_bias)
        self._cached = False # Unused, for the interface
        if config.use_spectral_norm:
            self.linear = spectral_norm(self.linear, name='weight', rescaling=config.weight_scale)

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        pass # Has no edge cache

    def forward(self, x):
        return self.linear(x)

def make_convolutions(input_dim: int, num_classes: int, cfg: ModelConfiguration, make_conv, *args, **kwargs):
    """ Makes convolutions from a class and a set of input, hidden and output dimensions. """
    all_dims = [input_dim] + list(cfg.hidden_sizes) + [num_classes]
    convs = []
    dims = list(zip(all_dims[:-1], all_dims[1:]))
    for idx, (in_dim, out_dim) in enumerate(dims):
        if idx == len(dims) - 1 and not cfg.use_spectral_norm_on_last_layer:
            sn = False
        else:
            sn = cfg.use_spectral_norm
        conv = make_conv(in_dim, out_dim, cfg, *args, **kwargs)
        if cfg.residual and idx < len(dims) - 1:
            conv = ResidualBlock(in_dim, out_dim, conv, cfg)
        convs.append(conv)
    return nn.ModuleList(convs)
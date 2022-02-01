from configuration import ModelConfiguration
import numpy as np
import torch.nn as nn
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from model.spectral_norm import spectral_norm
import pytorch_lightning as pl
import model.constants as mconst
from configuration import ModelConfiguration
from typing import Callable
from util import aggregate_matching

from model.nn import *



def _get_convolution_weights(conv: nn.Module) -> torch.Tensor:
    """ Gets the weight matrix of a convolution. """
    if isinstance(conv, ResidualBlock):
        return _get_convolution_weights(conv.conv)
    if isinstance(conv, GCNConv):
        return conv.lin.weight.detach()
    elif isinstance(conv, LinearWithSpectralNormaliatzion):
        return conv.linear.weight.detach()
    else:
        raise NotImplementedError(f'Cant get weights for convolution of type {type(conv)}')

class GCN(nn.Module):
    """ Vanilla GCN """

    def __init__(self, input_dim, num_classes, cfg: ModelConfiguration):
        super().__init__()
        self.activation = make_activation_by_configuration(cfg)
        self.residual = cfg.residual
        self.dropout = cfg.dropout
        self.drop_edge = cfg.drop_edge
        self.convs = make_convolutions(input_dim, num_classes, cfg, self._make_gcn_conv_with_spectral_norm)

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        for conv in self.convs:
            conv.clear_and_disable_cache()

    @staticmethod
    def _make_gcn_conv_with_spectral_norm(input_dim, output_dim, config: ModelConfiguration, *args, **kwargs):
        """ Method to create a convolution operator with spectral normalization """
        conv = GCNConv(input_dim, output_dim, *args, **kwargs, add_self_loops=False, cached=config.cached)
        if config.use_spectral_norm:
            conv.lin = spectral_norm(conv.lin, name='weight', rescaling=config.weight_scale)
        return conv

    def forward(self, data, dropout=True):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        if self.drop_edge > 0:
            edge_index, edge_weight = dropout_adj(edge_index, edge_attr=edge_weight, p=self.drop_edge, 
                                            force_undirected=False, training=dropout)
        embeddings = [x]
        for num, layer in enumerate(self.convs):
            x = layer(x, edge_index, edge_weight=edge_weight)
            if num < len(self.convs) - 1:
                x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, inplace=False, training=dropout)
            embeddings.append(x)
        return embeddings

    def get_output_weights(self):
        """ Gets the weights of the output layer. """
        return _get_convolution_weights(self.convs[-1])

# class GAT(nn.Module):
#     """ Graph Attention Network """

#     def __init__(self, input_dim, num_classes, hidden_dims, num_heads, activation=F.leaky_relu, 
#                     use_bias=True, use_spectral_norm=True, weight_scale=1.0,):
#         super().__init__()
#         self.activation = activation

#         self.convs = make_convolutions(input_dim, num_classes, hidden_dims, torch_geometric.nn.GATConv, num_heads,
#                                                 bias=use_bias, concat=False)
#         if use_spectral_norm:
#             for layer in self.convs:
#                 layer.lin_src = spectral_norm(layer.lin_src, name='weight', rescaling=weight_scale)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         embeddings = [x]
#         for num, layer in enumerate(self.convs):
#             x = layer(x, edge_index)
#             if num < len(self.convs) - 1:
#                 x = self.activation(x)
#             embeddings.append(x)
#         return embeddings

# class GraphSAGE(nn.Module):
#     """ GraphSAGE network. """

#     def __init__(self, input_dim, num_classes, hidden_dims, activation=F.leaky_relu, 
#                     use_bias=True, use_spectral_norm=True, weight_scale=1.0, normalize=False):
#         super().__init__()
#         self.activation = activation

#         self.convs = make_convolutions(input_dim, num_classes, hidden_dims, torch_geometric.nn.SAGEConv,
#                                                 bias=use_bias, normalize=normalize)
#         if use_spectral_norm:
#             for layer in self.convs:
#                 layer.lin_l = spectral_norm(layer.lin_l, name='weight', rescaling=weight_scale)
#                 layer.lin_r = spectral_norm(layer.lin_r, name='weight', rescaling=weight_scale)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         embeddings = [x]
#         for num, layer in enumerate(self.convs):
#             x = layer(x, edge_index)
#             if num < len(self.convs) - 1:
#                 x = self.activation(x)
#             embeddings.append(x)
#         return embeddings

# class GIN(nn.Module):
#     """ GIN network. """

#     def __init__(self, input_dim, num_classes, hidden_dims, activation=F.leaky_relu, 
#                     use_bias=True, use_spectral_norm=True, weight_scale=1.0):
#         super().__init__()
#         self.activation = activation

#         # Helper to build a GIN layer with spectral norm applied to its feature transformation module (single linear layer)
#         def GINConv_with_spectral_norm(in_dim, out_dim, bias=True, use_spectral_norm=True, weight_scale=1.0):
#             linear = LinearWithSpectralNormaliatzion(in_dim, out_dim, use_bias=bias, use_spectral_norm=use_spectral_norm, weight_scale=weight_scale)
#             return torch_geometric.nn.GINConv(linear)

#         self.convs = make_convolutions(input_dim, num_classes, hidden_dims, GINConv_with_spectral_norm,
#                                                 bias=use_bias, use_spectral_norm=use_spectral_norm, weight_scale=weight_scale)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         embeddings = [x]
#         for num, layer in enumerate(self.convs):
#             x = layer(x, edge_index)
#             if num < len(self.convs) - 1:
#                 x = self.activation(x)
#             embeddings.append(x)
#         return embeddings

# class MLP(nn.Module):
#     """ MLP on features. """

#     def __init__(self, input_dim, num_classes, hidden_dims, activation=F.leaky_relu, 
#                     use_bias=True, use_spectral_norm=True, weight_scale=1.0):
#         super().__init__()
#         self.activation = activation

#         self.convs = make_convolutions(input_dim, num_classes, hidden_dims, LinearWithSpectralNormaliatzion,
#                                                 use_bias=use_bias, use_spectral_norm=use_spectral_norm, weight_scale=weight_scale)

#     def forward(self, data):
#         x = data.x
#         embeddings = [x]
#         for num, layer in enumerate(self.convs):
#             x = layer(x)
#             if num < len(self.convs) - 1:
#                 x = self.activation(x)
#             embeddings.append(x)
#         return embeddings

class APPNP(nn.Module):
    """ Approximate Page Rank diffusion after MLP feature transformation. """

    def __init__(self, input_dim, num_classes, cfg: ModelConfiguration,):
        super().__init__()
        self.activation = make_activation_by_configuration(cfg)
        self.convs = make_convolutions(input_dim, num_classes, cfg, LinearWithSpectralNormaliatzion)
        self.appnp = APPNPConv(cfg.diffusion_iterations, cfg.teleportation_probability, cached=cfg.cached)

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        self.appnp.clear_and_disable_cache()

    def forward(self, data, store_diffused_features=True):
        x, edge_index = data.x, data.edge_index
        embeddings = [x]
        for num, layer in enumerate(self.convs):
            x = layer(x)
            if num < len(self.convs) - 1:
                x = self.activation(x)
                if store_diffused_features:
                    embeddings.append(self.appnp(x, edge_index))
                else:
                    embeddings.append(x)
            else:
                # Don't append the undiffused logits to `embeddings`, as otherwise they will be interpreted as features
                # TODO: Fix that, by not having -2 as the default layer for feature space, but instead make it model dependent
                pass

        # Diffusion
        x = self.appnp(x, edge_index)
        embeddings.append(x)
        return embeddings

    def get_output_weights(self):
        """ Gets the weights of the output layer. """
        return _get_convolution_weights(self.convs[-1])

def make_activation_by_configuration(configuration: ModelConfiguration):
    """ Makes the activation function form a configuration dict. """
    if configuration.activation == mconst.LEAKY_RELU:
        return nn.LeakyReLU()
    elif configuration.activation == mconst.RELU:
        return nn.ReLU()
    else:
        raise ValueError(f'Unsupported activation function {configuration.activation}')

def make_model_by_configuration(cfg: ModelConfiguration, input_dim, output_dim):
    """ Makes a gnn model function form a configuration dict. 
    
    Parameters:
    -----------
    configuration : dict
        Configuration of the model to make.
    input_dim : int
        Dimensionality of feature space the model works on.
    output_dim : int
        Output dimensionality (e.g. number of classes) the model produces.
    
    Returns:
    --------
    model : torch.nn.Module
        A torch model that takes a torch_geometric.data.Batch as an input and outputs a list that correpsond to embeddings in all layers.
    """
    if cfg.model_type == mconst.GCN:
        return GCN(input_dim, output_dim, cfg)
    # elif configuration['model_type'] == 'gat':
    #     return GAT(input_dim, output_dim, configuration['hidden_sizes'], configuration['num_heads'], make_activation_by_configuration(configuration), 
    #         use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'])
    # elif configuration['model_type'] == 'sage':
    #     return GraphSAGE(input_dim, output_dim, configuration['hidden_sizes'], make_activation_by_configuration(configuration), 
    #         use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'],
    #         normalize=configuration['normalize'])
    # elif configuration['model_type'] == 'gin':
    #     return GIN(input_dim, output_dim, configuration['hidden_sizes'], make_activation_by_configuration(configuration), 
    #         use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'])
    # elif configuration['model_type'] == 'mlp':
    #     return MLP(input_dim, output_dim, configuration['hidden_sizes'], make_activation_by_configuration(configuration), 
    #         use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'])
    elif cfg.model_type == mconst.APPNP:
        return APPNP(input_dim, output_dim, cfg)
    else:
        raise RuntimeError(f'Unsupported model type {cfg.model_type}')
        
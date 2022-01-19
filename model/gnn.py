from configuration import ModelConfiguration
import numpy as np
import torch.nn as nn
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from model.spectral_norm import spectral_norm
import pytorch_lightning as pl
from metrics import accuracy
from scipy.stats import ortho_group
import model.constants as mconst
from configuration import ModelConfiguration

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

    def __init__(self, input_dim, output_dim, conv, use_spectral_norm=False, weight_scale=1.0, use_bias=False,
        freeze_residual_projection=False, orthogonalize_residual_projection=False):
        super().__init__()
        self.conv = conv
        if input_dim != output_dim:
            self.input_projection = LinearWithSpectralNormaliatzion(input_dim, output_dim, use_bias=use_bias,
                use_spectral_norm=use_spectral_norm, weight_scale=weight_scale)

            if orthogonalize_residual_projection:
                raise NotImplementedError('Currently orthogonalization of projection matrices is not yet supported.')
            
            if freeze_residual_projection:
                for param in self.input_projection.parameters():
                    param.requires_grad = False
        else:
            self.input_projection = None
        
    def forward(self, x, edge_index, **kwargs):
        h = self.conv(x, edge_index, **kwargs)
        if self.input_projection:
            x = self.input_projection(x)
        return x + h

def _make_convolutions(input_dim, num_classes, cfg: ModelConfiguration, make_conv, *args, **kwargs):
    """ Makes convolutions from a class and a set of input, hidden and output dimensions. """
    all_dims = [input_dim] + list(cfg.hidden_sizes) + [num_classes]
    convs = []
    dims = list(zip(all_dims[:-1], all_dims[1:]))
    for idx, (in_dim, out_dim) in enumerate(dims):
        if idx == len(dims) - 1 and not cfg.use_spectral_norm_on_last_layer:
            sn = False
        else:
            sn = cfg.use_spectral_norm
        conv = make_conv(in_dim, out_dim, *args, use_spectral_norm=sn, 
            weight_scale=cfg.weight_scale, **kwargs)
        if cfg.residual:
            conv = ResidualBlock(in_dim, out_dim, conv, use_spectral_norm=sn, weight_scale=cfg.weight_scale,
                freeze_residual_projection=cfg.freeze_residual_projection, orthogonalize_residual_projection=False)
        convs.append(conv)
    return nn.ModuleList(convs)

class LinearWithSpectralNormaliatzion(nn.Module):
    """ Wrapper for a linear layer that applies spectral normalization and rescaling to the weight. """

    def __init__(self, input_dim, output_dim, use_bias=True, use_spectral_norm=False, weight_scale=1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        if use_spectral_norm:
            self.linear = spectral_norm(self.linear, name='weight', rescaling=weight_scale)
        
    def forward(self, x):
        return self.linear(x)

class GCN(nn.Module):
    """ Vanilla GCN """

    def __init__(self, input_dim, num_classes, cfg: ModelConfiguration):
        super().__init__()
        self.activation = make_activation_by_configuration(cfg)
        self.residual = cfg.residual
        self.dropout = cfg.dropout
        self.drop_edge = cfg.drop_edge
        self.convs = _make_convolutions(input_dim, num_classes, cfg, self._make_conv_with_spectral_norm)

    @staticmethod
    def _make_conv_with_spectral_norm(input_dim, output_dim, *args, use_spectral_norm=False, weight_scale=1.0, **kwargs):
        conv = torch_geometric.nn.GCNConv(input_dim, output_dim, *args, **kwargs, add_self_loops=False)
        if use_spectral_norm:
            conv.lin = spectral_norm(conv.lin, name='weight', rescaling=weight_scale)
        return conv

    def forward(self, data, dropout=True):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        if self.drop_edge > 0:
            edge_index, edge_weight = dropout_adj(edge_index, edge_attr=edge_weight, p=self.drop_edge, 
                                            force_undirected=False, training=dropout)
        embeddings = []
        for num, layer in enumerate(self.convs):
            x = layer(x, edge_index, edge_weight=edge_weight)
            if num < len(self.convs) - 1:
                x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, inplace=False, training=dropout)
            embeddings.append(x)
        return embeddings

class GAT(nn.Module):
    """ Graph Attention Network """

    def __init__(self, input_dim, num_classes, hidden_dims, num_heads, activation=F.leaky_relu, 
                    use_bias=True, use_spectral_norm=True, weight_scale=1.0,):
        super().__init__()
        self.activation = activation

        self.convs = _make_convolutions(input_dim, num_classes, hidden_dims, torch_geometric.nn.GATConv, num_heads,
                                                bias=use_bias, concat=False)
        if use_spectral_norm:
            for layer in self.convs:
                layer.lin_src = spectral_norm(layer.lin_src, name='weight', rescaling=weight_scale)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embeddings = []
        for num, layer in enumerate(self.convs):
            x = layer(x, edge_index)
            if num < len(self.convs) - 1:
                x = self.activation(x)
            embeddings.append(x)
        return embeddings

class GraphSAGE(nn.Module):
    """ GraphSAGE network. """

    def __init__(self, input_dim, num_classes, hidden_dims, activation=F.leaky_relu, 
                    use_bias=True, use_spectral_norm=True, weight_scale=1.0, normalize=False):
        super().__init__()
        self.activation = activation

        self.convs = _make_convolutions(input_dim, num_classes, hidden_dims, torch_geometric.nn.SAGEConv,
                                                bias=use_bias, normalize=normalize)
        if use_spectral_norm:
            for layer in self.convs:
                layer.lin_l = spectral_norm(layer.lin_l, name='weight', rescaling=weight_scale)
                layer.lin_r = spectral_norm(layer.lin_r, name='weight', rescaling=weight_scale)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embeddings = []
        for num, layer in enumerate(self.convs):
            x = layer(x, edge_index)
            if num < len(self.convs) - 1:
                x = self.activation(x)
            embeddings.append(x)
        return embeddings

class GIN(nn.Module):
    """ GIN network. """

    def __init__(self, input_dim, num_classes, hidden_dims, activation=F.leaky_relu, 
                    use_bias=True, use_spectral_norm=True, weight_scale=1.0):
        super().__init__()
        self.activation = activation

        # Helper to build a GIN layer with spectral norm applied to its feature transformation module (single linear layer)
        def GINConv_with_spectral_norm(in_dim, out_dim, bias=True, use_spectral_norm=True, weight_scale=1.0):
            linear = LinearWithSpectralNormaliatzion(in_dim, out_dim, use_bias=bias, use_spectral_norm=use_spectral_norm, weight_scale=weight_scale)
            return torch_geometric.nn.GINConv(linear)

        self.convs = _make_convolutions(input_dim, num_classes, hidden_dims, GINConv_with_spectral_norm,
                                                bias=use_bias, use_spectral_norm=use_spectral_norm, weight_scale=weight_scale)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embeddings = []
        for num, layer in enumerate(self.convs):
            x = layer(x, edge_index)
            if num < len(self.convs) - 1:
                x = self.activation(x)
            embeddings.append(x)
        return embeddings

class MLP(nn.Module):
    """ MLP on features. """

    def __init__(self, input_dim, num_classes, hidden_dims, activation=F.leaky_relu, 
                    use_bias=True, use_spectral_norm=True, weight_scale=1.0):
        super().__init__()
        self.activation = activation

        self.convs = _make_convolutions(input_dim, num_classes, hidden_dims, LinearWithSpectralNormaliatzion,
                                                use_bias=use_bias, use_spectral_norm=use_spectral_norm, weight_scale=weight_scale)

    def forward(self, data):
        x = data.x
        embeddings = []
        for num, layer in enumerate(self.convs):
            x = layer(x)
            if num < len(self.convs) - 1:
                x = self.activation(x)
            embeddings.append(x)
        return embeddings

class APPNP(nn.Module):
    """ Approximate Page Rank diffusion after MLP feature transformation. """

    def __init__(self, input_dim, num_classes, hidden_dims, activation=F.leaky_relu, 
                    use_bias=True, use_spectral_norm=True, weight_scale=1.0, 
                    diffusion_iterations=2, teleportation_probability=0.1, cached=True):
        super().__init__()
        self.activation = activation

        self.convs = _make_convolutions(input_dim, num_classes, hidden_dims, LinearWithSpectralNormaliatzion,
                                                use_bias=use_bias, use_spectral_norm=use_spectral_norm, weight_scale=weight_scale,)
        self.appnp = torch_geometric.nn.APPNP(diffusion_iterations, teleportation_probability, cached=cached)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embeddings = []
        for num, layer in enumerate(self.convs):
            x = layer(x)
            if num < len(self.convs) - 1:
                x = self.activation(x)
            embeddings.append(x)

        # Diffusion
        x = self.appnp(x, edge_index)
        embeddings.append(x)

        return embeddings

def make_activation_by_configuration(configuration: ModelConfiguration):
    """ Makes the activation function form a configuration dict. """
    if configuration.activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif configuration.activation == 'relu':
        return nn.ReLU()
    else:
        raise RuntimeError(f'Unsupported activation function {configuration.activation}')

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
    if cfg.model_type in mconst.GCN:
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
    # elif configuration['model_type'] == 'appnp':
    #     return APPNP(input_dim, output_dim, configuration['hidden_sizes'], make_activation_by_configuration(configuration), 
    #         use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'],
    #         diffusion_iterations=configuration['diffusion_iterations'], teleportation_probability=configuration['teleportation_probability'],)
    else:
        raise RuntimeError(f'Unsupported model type {configuration["model_type"]}')
        
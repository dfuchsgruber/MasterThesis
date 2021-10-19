import numpy as np
import torch.nn as nn
import torch
import torch_geometric
import torch.nn.functional as F
from model.spectral_norm import spectral_norm
import pytorch_lightning as pl
from metrics import accuracy

def _make_convolutions(input_dim, num_classes, hidden_dims, convolution_class, *args, **kwargs):
    all_dims = [input_dim] + list(hidden_dims) + [num_classes]
    convs = []
    return nn.ModuleList([
       convolution_class(in_dim, out_dim, *args, **kwargs) for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:])
    ])

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

    def __init__(self, input_dim, num_classes, hidden_dims, activation=F.leaky_relu, 
                    use_bias=True, use_spectral_norm=True, weight_scale=1.0, cached=True,):
        super().__init__()
        self.activation = activation

        self.convs = _make_convolutions(input_dim, num_classes, hidden_dims, torch_geometric.nn.GCNConv,
                                                cached=cached, bias=use_bias)
        if use_spectral_norm:
            for layer in self.convs:
                layer.lin = spectral_norm(layer.lin, name='weight', rescaling=weight_scale)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embeddings = []
        for num, layer in enumerate(self.convs):
            x = layer(x, edge_index)
            if num < len(self.convs) - 1:
                x = self.activation(x)
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

def make_activation_by_configuration(configuration):
    """ Makes the activation function form a configuration dict. """
    if configuration['activation'] == 'leaky_relu':
        return nn.LeakyReLU()
    elif configuration['activation'] == 'relu':
        return nn.ReLU()
    else:
        raise RuntimeError(f'Unsupported activation function {configuration["activation"]}')

def make_model_by_configuration(configuration, input_dim, output_dim):
    """ Makes a gnn model function form a configuration dict. """
    if configuration['model_type'] == 'gcn':
        return GCN(input_dim, output_dim, configuration['hidden_sizes'], make_activation_by_configuration(configuration), 
            use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'],
            cached=True)
    elif configuration['model_type'] == 'gat':
        return GAT(input_dim, output_dim, configuration['hidden_sizes'], configuration['num_heads'], make_activation_by_configuration(configuration), 
            use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'])
    elif configuration['model_type'] == 'sage':
        return GraphSAGE(input_dim, output_dim, configuration['hidden_sizes'], make_activation_by_configuration(configuration), 
            use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'],
            normalize=configuration['normalize'])
    elif configuration['model_type'] == 'gin':
        return GIN(input_dim, output_dim, configuration['hidden_sizes'], make_activation_by_configuration(configuration), 
            use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'])
    elif configuration['model_type'] == 'mlp':
        return MLP(input_dim, output_dim, configuration['hidden_sizes'], make_activation_by_configuration(configuration), 
            use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'])
    elif configuration['model_type'] == 'appnp':
        return APPNP(input_dim, output_dim, configuration['hidden_sizes'], make_activation_by_configuration(configuration), 
            use_bias=configuration['use_bias'], use_spectral_norm=configuration['use_spectral_norm'], weight_scale=configuration['weight_scale'],
            diffusion_iterations=configuration['diffusion_iterations'], teleportation_probability=configuration['teleportation_probability'],)
    else:
        raise RuntimeError(f'Unsupported model type {configuration["model_type"]}')
        
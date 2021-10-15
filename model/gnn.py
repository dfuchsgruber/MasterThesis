import numpy as np
import torch.nn as nn
import torch
import torch_geometric
import torch.nn.functional as F
from model.spectral_norm import spectral_norm

class GCNConv(nn.Module):
    """ Wrapper for a vanilla GCN convolution. """

    def __init__(self, input_dim, output_dim, use_bias=True, use_spectral_norm=False, upper_lipschitz_bound=1.0):
        super().__init__()
        self.conv = torch_geometric.nn.GCNConv(input_dim, output_dim, bias=use_bias, cached=True)
        # Apply spectral norm to the linear layer of the GCN conv
        if use_spectral_norm:
            self.conv.lin = spectral_norm(self.conv.lin, name='weight', rescaling=upper_lipschitz_bound)

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight=edge_weight) 

class GATConv(nn.Module):
    """ Wrapper for a GAT convolution. """

    def __init__(self, input_dim, output_dim, num_heads=2, use_bias=True, use_spectral_norm=False, upper_lipschitz_bound=1.0):
        super().__init__()
        self.conv = torch_geometric.nn.GATConv(input_dim, output_dim, num_heads, concat=False, bias=use_bias)
        if use_spectral_norm:
            self.conv.lin_src = spectral_norm(self.conv.lin_src, name='weight', rescaling=upper_lipschitz_bound)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index) 
        return x

class SAGEConv(nn.Module):
    """ Wrapper for a GraphSAGE convolution. """

    def __init__(self, input_dim, output_dim, use_bias=True, use_spectral_norm=False, upper_lipschitz_bound=1.0, normalize=True):
        super().__init__()
        self.conv = torch_geometric.nn.SAGEConv(input_dim, output_dim, bias=use_bias, normalize=normalize)
        # Apply spectral norm to both linear transformations in the SAGE layer
        if use_spectral_norm:
            self.conv.lin_l = spectral_norm(self.conv.lin_l, name='weight', rescaling=upper_lipschitz_bound)
            self.conv.lin_r = spectral_norm(self.conv.lin_r, name='weight', rescaling=upper_lipschitz_bound)

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index) 

class GINConv(nn.Module):
    """ Wrapper for a GIN convolution. """

    def __init__(self, input_dim, output_dim, use_bias=True, use_spectral_norm=False, upper_lipschitz_bound=1.0, normalize=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        if use_spectral_norm:
            self.linear = spectral_norm(self.linear, name='weight', rescaling=upper_lipschitz_bound)
        self.conv = torch_geometric.nn.GINConv(self.linear)

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index) 

class LinearWithSpectralNormaliatzion(nn.Module):
    """ Wrapper for a linear layer that applies spectral normalization and rescaling to the weight. """

    def __init__(self, input_dim, output_dim, use_bias=True, use_spectral_norm=False, upper_lipschitz_bound=1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        if use_spectral_norm:
            self.linear = spectral_norm(self.linear, name='weight', rescaling=upper_lipschitz_bound)
        
    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

class GNN(nn.Module):
    """ Wrapper module for different GNN layer types. """

    def __init__(self, layer_type, input_dim, layer_dims, num_classes, 
        use_bias=True, use_spectral_norm=True, activation='leaky_relu', leaky_relu_slope=0.01,
        upper_lipschitz_bound=1.0, num_heads=1, teleportation_probability=0.1, diffusion_iterations=3):
        """ Builds a general GNN with a specified layer type. 
        
        Parameters:
        -----------
        layer_type : ('gcn', 'gat', 'gin', 'appnp', 'mlp', 'sage')
            Which layer to use in the GNN.
        input_dim : int
            Number of input features.
        layer_dims : iterable of ints
            Each value depicts a hidden layer.
        num_classes : int
            How many output classes to give.
        use_bias : bool
            If layers have bias.
        use_spectral_norm : bool
            If spectral normalization is applied to the weights of each layer.
        activation : ('relu', 'leaky_relu')
            Activation function after each layer.
        leaky_relu_slope : float
            Slope of the leaky relu if used.
        upper_lipschitz_bound : float
            If spectral normalization is used, re-scales the weight matrix. This induces a new upper_lipschitz_bound.
        num_heads : int
            How many heads are used for multi-head convolutions like GAT.
        teleportation_probability : float
            Probability of teleportation in APPNP.
        diffusion_iterations : int
            How many diffusion iterations are used to approximate PPR in APPNP.
        """
        super().__init__()
        all_dims = [input_dim] + list(layer_dims) + [num_classes]
        self.use_spectral_norm = use_spectral_norm
        self.use_bias = use_bias
        self.activation = activation
        self.layer_type = layer_type
        self.leaky_relu_slope = leaky_relu_slope
        self.upper_lipschitz_bound = upper_lipschitz_bound
        self.num_heads = num_heads
        self.teleportation_probability = teleportation_probability
        self.diffusion_iterations = diffusion_iterations

        if self.layer_type == 'gcn':
            make_layer = lambda in_dim, out_dim: GCNConv(in_dim, out_dim, use_bias=self.use_bias, use_spectral_norm=self.use_spectral_norm, 
                upper_lipschitz_bound=self.upper_lipschitz_bound)
        elif self.layer_type == 'gat':
            make_layer = lambda in_dim, out_dim: GATConv(in_dim, out_dim, use_bias=self.use_bias, use_spectral_norm=self.use_spectral_norm, 
                upper_lipschitz_bound=self.upper_lipschitz_bound, num_heads = self.num_heads)
        elif self.layer_type == 'sage':
            make_layer = lambda in_dim, out_dim: SAGEConv(in_dim, out_dim, use_bias=self.use_bias, use_spectral_norm=self.use_spectral_norm, 
                upper_lipschitz_bound=self.upper_lipschitz_bound,)
        elif self.layer_type == 'appnp': # A bit hacky, since technically this isn't a layer.
            # A MLP is used to preprocess the features.
            make_layer = lambda in_dim, out_dim: LinearWithSpectralNormaliatzion(in_dim, out_dim, use_bias=self.use_bias, use_spectral_norm=self.use_spectral_norm,
                upper_lipschitz_bound=self.upper_lipschitz_bound)
            self.appnp = torch_geometric.nn.APPNP(self.diffusion_iterations, self.teleportation_probability, cached=True)
        elif self.layer_type == 'mlp':
            make_layer = lambda in_dim, out_dim: LinearWithSpectralNormaliatzion(in_dim, out_dim, use_bias=self.use_bias, use_spectral_norm=self.use_spectral_norm,
                upper_lipschitz_bound=self.upper_lipschitz_bound)
        elif self.layer_type == 'gin':
            make_layer = lambda in_dim, out_dim: GINConv(in_dim, out_dim, use_bias=self.use_bias, use_spectral_norm=self.use_spectral_norm,
                upper_lipschitz_bound=self.upper_lipschitz_bound,)
        else:
            raise RuntimeError(f'Unsupported layer type {self.layer_type}')
        self.layers = torch.nn.ModuleList([
            make_layer(in_dim, out_dim) for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:])
        ])

    def forward(self, x, edge_index):
        """ Forward pass through the network. 
        
        Parameters:
        -----------
        x : torch.Tensor, shape [N, input_dim]
            Attribute matrix for vertices.
        edge_index : torch.Tensor, shape [2, E]
            Edge indices for the graph.
        
        Returns:
        --------
        embeddings : list
            Embeddings after each layer in the network.
            Note that embeddings[-1] are the model logits.
        """
        embeddings = []
        for num, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if num < len(self.layers) - 1:
                if self.activation == 'leaky_relu':
                    x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
                elif self.activation == 'relu':
                    x = F.relu(x)
                else:
                    raise RuntimeError(f'Unsupported activation type {self.activation}')
            embeddings.append(x)
        if self.layer_type == 'appnp':
            x = self.appnp(x, edge_index)
            embeddings.append(x)
        return embeddings

if __name__ == '__main__':
    model = GNN('gcn', 10, [16, 16], 3, use_spectral_norm=True)
    x_test = torch.rand([4, 10])
    edge_index = torch.tensor([[0, 1, 2, 3, 1, 2, 3], [1, 2, 3, 0, 0, 0, 1]])
    print(model(x_test, edge_index))
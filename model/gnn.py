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
        self.conv = torch_geometric.nn.GCNConv(input_dim, output_dim, bias=use_bias)
        # Apply spectral norm to the linear layer of the GCN conv
        if use_spectral_norm:
            self.conv.lin = spectral_norm(self.conv.lin, name='weight', rescaling=upper_lipschitz_bound)


    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight=edge_weight) 

class GNN(nn.Module):
    """ Wrapper module for different GNN layer types. """

    def __init__(self, layer_type, input_dim, layer_dims, num_classes, 
        use_bias=True, use_spectral_norm=True, activation='leaky_relu', leaky_relu_slope=0.01,
        upper_lipschitz_bound=1.0):
        """ Builds a general GNN with a specified layer type. 
        
        Parameters:
        -----------
        layer_type : ('gcn', 'gat', 'gin', 'appnp')
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
        """
        super().__init__()
        all_dims = [input_dim] + list(layer_dims) + [num_classes]
        self.use_spectral_norm = use_spectral_norm
        self.use_bias = use_bias
        self.activation = activation
        self.layer_type = layer_type
        self.leaky_relu_slope = leaky_relu_slope
        self.upper_lipschitz_bound = upper_lipschitz_bound

        if self.layer_type == 'gcn':
            make_layer = lambda in_dim, out_dim: GCNConv(in_dim, out_dim, use_bias=self.use_bias, use_spectral_norm=self.use_spectral_norm, 
                upper_lipschitz_bound=self.upper_lipschitz_bound)
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
        return embeddings

if __name__ == '__main__':
    model = GNN('gcn', 10, [16, 16], 3, use_spectral_norm=True)
    x_test = torch.rand([4, 10])
    edge_index = torch.tensor([[0, 1, 2, 3, 1, 2, 3], [1, 2, 3, 0, 0, 0, 1]])
    print(model(x_test, edge_index))
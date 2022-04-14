from copy import deepcopy
from configuration import ModelConfiguration
import numpy as np
import torch.nn as nn
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from model.parametrization import spectral_norm, bjorck_orthonormalize
import pytorch_lightning as pl
import model.constants as mconst
from configuration import ModelConfiguration
from typing import Callable
from util import aggregate_matching
import laplace

from model.nn import *
from model.prediction import *
from model.bgcn import BayesianGCNConv

def _get_convolution_weights(conv: nn.Module) -> torch.Tensor:
    """ Gets the weight matrix of a convolution. """
    if isinstance(conv, [ResidualBlock, BasicBlock]):
        return _get_convolution_weights(conv.conv)
    if isinstance(conv, GCNConv):
        return conv.lin.weight.detach()
    elif isinstance(conv, LinearWithParametrization):
        return conv.linear.weight.detach()
    elif isinstance(conv, BaysianGCNConv):
        return _get_convolution_weights(conv.lin)
    elif isinstance(conv, BayesianLinear):
        return conv.w_mu.detach()
    else:
        raise NotImplementedError(f'Cant get weights for convolution of type {type(conv)}')

class ModelBase(nn.Module):
    """ Base class for GNN models. """

    training_type = mconst.TRAIN_PL

    def __init__(self, config: ModelConfiguration):
        super().__init__()
        self.drop_edge = config.drop_edge

    def _unpack_inputs_and_sample(self, data, sample=None):
        """ Applies dropout for edges if applicable and returns if the model should sample. """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        if sample is None:
            sample = self.training
        if self.drop_edge > 0:
            edge_index, edge_weight = dropout_adj(edge_index, edge_attr=edge_weight, p=self.drop_edge, 
                                            force_undirected=False, training=sample)
        return x, edge_index, edge_weight, sample

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        raise NotImplemented

    def get_output_weights(self):
        """ Gets the weights of the output layer. """
        raise NotImplemented

    def losses(self, prediction: Prediction) -> Dict:
        """ Gets additional losses based on the model parameters. """
        return {}

class GCN(ModelBase):
    """ Vanilla GCN """

    def __init__(self, input_dim, num_classes, cfg: ModelConfiguration):
        super().__init__(cfg)
        self.residual = cfg.residual
        self.convs = make_convolutions(input_dim, num_classes, cfg, self._make_conv)

    def get_weights(self) -> Dict[str, nn.Parameter]:
        """ Gets all parameters of the model. """
        params = {}
        for idx, conv in enumerate(self.convs):
            params |= {
                f'conv{idx}.{name}' : param for name, param in conv.get_weights().items()
            }
        return params

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        for conv in self.convs:
            conv.clear_and_disable_cache()

    @staticmethod
    def _make_conv(input_dim, output_dim, config: ModelConfiguration, *args, 
            use_spectral_norm: bool = False, 
            use_bjorck_norm: bool=False, 
            use_forbenius_norm: bool=False, 
            use_rescaling: bool=False, **kwargs):
        """ Method to create a convolution operator with spectral normalization """
        return GCNConv(input_dim, output_dim, config, *args, **kwargs, use_spectral_norm=use_spectral_norm, use_bjorck_norm=use_bjorck_norm,
            use_rescaling=use_rescaling, use_forbenius_norm=use_forbenius_norm, add_self_loops=False, bias=config.use_bias)

    def forward(self, data, sample=None):
        # print(data.x.device, data.edge_index.device, data.edge_weight.device)
        x, edge_index, edge_weight, sample = self._unpack_inputs_and_sample(data, sample=sample)
        embeddings = [x]
        for num, layer in enumerate(self.convs):
            # print(num, x.device, edge_index.device, edge_weight.device)
            # print(num, edge_index.size())
            x = layer(x, edge_index, edge_weight=edge_weight, sample=sample)
            embeddings.append(x)
        return Prediction(
            features = embeddings,
            **{
                LOGITS : x,
                SOFT_PREDICTIONS : F.softmax(x, dim=1),
                INPUTS : data.x,
            } 
            )

    def get_output_weights(self):
        """ Gets the weights of the output layer. """
        return _get_convolution_weights(self.convs[-1])

class GAT(GCN):
    """ Vanilla GAT """

    @staticmethod
    def _make_conv(input_dim, output_dim, config: ModelConfiguration, *args, 
            use_spectral_norm: bool = False, 
            use_bjorck_norm: bool=False, 
            use_forbenius_norm: bool=False, 
            use_rescaling: bool=False, **kwargs):
        """ Method to create a convolution operator with spectral normalization """
        return GATConv(input_dim, output_dim, config, *args, **kwargs, use_spectral_norm=use_spectral_norm, use_bjorck_norm=use_bjorck_norm,
            use_rescaling=use_rescaling, use_forbenius_norm=use_forbenius_norm, add_self_loops=False, bias=config.use_bias)

class MLP(GCN):
    """ Vanilla MLP. """

    @staticmethod
    def _make_conv(input_dim, output_dim, config: ModelConfiguration, *args, 
            use_spectral_norm: bool = False, 
            use_bjorck_norm: bool=False, 
            use_forbenius_norm: bool=False, 
            use_rescaling: bool=False, **kwargs):
        """ Method to create a convolution operator with spectral normalization """
        return LinearWithParametrization(input_dim, output_dim, config, *args, **kwargs, use_spectral_norm=use_spectral_norm, use_bjorck_norm=use_bjorck_norm,
            use_rescaling=use_rescaling, use_forbenius_norm=use_forbenius_norm)


class GCNLinearClassification(GCN):
    """ Vanilla GCN encoder with linear classification output """

    def __init__(self, input_dim, num_classes, cfg: ModelConfiguration):
        super(GCN, self).__init__(cfg)

        # Hacky workaround: Create one additional convolution that will then be replaced with a linear layer
        cfg_convs = deepcopy(cfg)
        cfg_convs.hidden_sizes = cfg.hidden_sizes[:-1] + [cfg.hidden_sizes[-1], cfg.hidden_sizes[-1]]
        self.convs = make_convolutions(input_dim, num_classes, cfg_convs, self._make_conv)[:-1]
        head_conv = LinearWithParametrization(
                cfg.hidden_sizes[-1], num_classes, cfg, use_spectral_norm=cfg.use_spectral_norm_on_last_layer,
                use_bjorck_norm=cfg.use_bjorck_norm_on_last_layer, use_forbenius_norm=cfg.use_bjorck_norm_on_last_layer,
                use_rescaling=cfg.use_rescaling,
            )

        if cfg.residual and cfg.use_residual_on_last_layer:
            self.head = ResidualBlock(cfg.hidden_sizes[-1], num_classes, head_conv, None, cfg, 
                use_spectral_norm=cfg.use_spectral_norm_on_last_layer, 
                use_bjorck_norm=cfg.use_bjorck_norm_on_last_layer, 
                use_forbenius_norm=cfg.use_bjorck_norm_on_last_layer,
                use_rescaling=cfg.use_rescaling_on_last_layer,
                )
        else:
            self.head = BasicBlock(cfg.hidden_sizes[-1], num_classes, head_conv, None, cfg)

    def get_weights(self) -> Dict[str, nn.Parameter]:
        return super().get_weights() | {
            f'head.{name}' : param for name, param in self.head.get_weights().items()
        }

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        super().clear_and_disable_cache()
        self.head.clear_and_disable_cache()
    
    def get_output_weights(self):
        """ Gets the weights of the output layer. """
        return _get_convolution_weights(self.head)

    def forward(self, data, sample=None):
        x, edge_index, edge_weight, sample = self._unpack_inputs_and_sample(data, sample=sample)
        embeddings = [x]
        for num, layer in enumerate(self.convs):
            x = layer(x, edge_index, edge_weight=edge_weight, sample=sample)
            embeddings.append(x)
        x = self.head(x, edge_index, edge_weight=edge_weight, sample=sample)
        embeddings.append(x)
        return Prediction(
            features = embeddings,
            **{
                LOGITS : x,
                SOFT_PREDICTIONS : F.softmax(x, dim=1),
                INPUTS : data.x,
            })

class GCNLinearClassificationLaplaceBackbone(GCNLinearClassification):
    """ Child-backbone class for laplace approximation. """
    training_type = mconst.TRAIN_LAPLACE

class GCNLaplace(ModelBase):
    """ GCN with linear output layer that has a laplace approximation to its weights. It needs to be assembled
    as an GCN encoder (ModuleList of convolutions) and a laplace linear layer. """

    def __init__(self, convs: nn.ModuleList, head: laplace.Laplace, config: ModelConfiguration):
        super().__init__(config)
        self.convs = convs
        self.head = head

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        for conv in self.convs:
            conv.clear_and_disable_cache()

    def get_output_weights(self):
        """ Gets the weights of the output layer. """
        return _get_convolution_weights(self.head)

    def forward(self, data, sample=None):
        x, edge_index, edge_weight, sample = self._unpack_inputs_and_sample(data, sample=sample)
        embeddings = [x]
        for num, layer in enumerate(self.convs):
            x = layer(x, edge_index, edge_weight=edge_weight, sample=sample)
            embeddings.append(x)
        if sample:
            # print(x.device)
            # print(x.device, {name : param.device for name, param in self.head.parameters()})
            # Always draw only one sample: The samples themselves will be aggregated by the framework 
            # in `semi_supervised_node_classification.py`
            x = self.head.predictive_samples(x, n_samples=1)[0, ...]
        else:
            x = self.head(x)
        embeddings.append(x)
        return Prediction(
            features = embeddings,
            **{
                LOGITS : x,
                SOFT_PREDICTIONS : F.softmax(x, dim=1),
                INPUTS : data.x,
            })


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
#             linear = LinearWithParametrization(in_dim, out_dim, use_bias=bias, use_spectral_norm=use_spectral_norm, weight_scale=weight_scale)
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

#         self.convs = make_convolutions(input_dim, num_classes, hidden_dims, LinearWithParametrization,
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


class BayesianGCN(ModelBase):
    """Bayesian GCN model"""

    def __init__(self, input_dim: int, output_dim: int, config: ModelConfiguration):
        super().__init__(config)
        self.activation = make_activation_by_configuration(config)
        self.convs = make_convolutions(input_dim, output_dim, config, self._make_variational_gcn_conv)
        self.q_weight = config.bgcn.q_weight
        self.prior_weight = config.bgcn.prior_weight

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        for conv in self.convs:
            conv.clear_and_disable_cache()

    @staticmethod
    def _make_variational_gcn_conv(input_dim, output_dim, config: ModelConfiguration, *args, **kwargs):
        """ Method to create a convolution. """
        conv = BayesianGCNConv(input_dim, output_dim, sigma_1 = config.bgcn.sigma_1, sigma_2 = config.bgcn.sigma_2, pi=config.bgcn.pi,
                                cached=config.cached, add_self_loops=False

        )
        return conv

    def forward(self, data: torch_geometric.data.Data, sample: bool=None) -> Prediction:
        x, edge_index, edge_weight, sample = self._unpack_inputs_and_sample(data, sample=sample)
        embeddings = [x]
        for num, layer in enumerate(self.convs):
            x = layer(x, edge_index, edge_weight=edge_weight, calculate_log_probs=True, sample=sample)
            embeddings.append(x)
        return Prediction(
            features = embeddings, 
            log_prior = self.log_prior(),
            log_q = self.log_q(),
            **{
                LOGITS : x,
                SOFT_PREDICTIONS : F.softmax(x, dim=1),
                INPUTS : data.x,
            },
            )

    def log_prior(self) -> torch.Tensor:
        return sum(conv.log_prior for conv in self.convs)

    def log_q(self) -> torch.Tensor:
        return sum(conv.log_q for conv in self.convs)

    def losses(self, prediction: Prediction) -> Dict[str, torch.Tensor]:
        return {
            'log_q' : self.q_weight * prediction.get('log_q'),
            'log_prior' : self.prior_weight * (-prediction.get('log_prior')),
        }


# class BGCN(ModelBase):
#     """ Bayesian GCN that samples weights. """

#     def __init__(self, input_dim, num_classes, cfg: ModelConfiguration):
#         super().__init__()
#         self.activation = make_activation_by_configuration(cfg)
#         self.residual = cfg.residual
#         self.dropout = cfg.dropout
#         self.drop_edge = cfg.drop_edge
#         self.convs = make_convolutions(input_dim, num_classes, cfg, self._make_bgcn_conv)
#         self.kl_loss_weight = cfg.kl_loss_weight

#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
#         embeddings = [x]
#         for num, layer in enumerate(self.convs):
#             x = layer(x) #, edge_index, edge_weight=edge_weight)
#             if num < len(self.convs) - 1:
#                 x = self.activation(x)
#             embeddings.append(x)
#         return Prediction(
#             embeddings, 
#             inputs = data.x,
#             logits = x,
#             soft = F.softmax(x, dim=1),
#             )

#     def clear_and_disable_cache(self):
#         """ Clears and disables the cache. """
#         for conv in self.convs:
#             conv.clear_and_disable_cache()

#     # def losses(self) -> Dict:
#     #     """ Gets additional losses based on the model parameters. """
#     #     num_kl_terms = sum(conv.num_kl_terms() for conv in self.convs)
#     #     return super().losses() | {
#     #         f'kl-divergence-layer-{idx}' : conv.kl_loss() * self.kl_loss_weight / num_kl_terms for idx, conv in enumerate(self.convs)
#     #     }

#     @staticmethod
#     def _make_bgcn_conv(input_dim, output_dim, config: ModelConfiguration, *args, **kwargs):
#         """ Method to create a convolution operator with spectral normalization """
#         # conv = BaysianGCNConv(input_dim, output_dim, *args, **kwargs, add_self_loops=False, cached=config.cached,
#         #     prior_mean_bias=config.prior_mean_bias, prior_variance_bias=config.prior_variance_bias,
#         #     prior_mean_weight=config.prior_mean_weight, prior_variance_weight=config.prior_variance_weight, bias=config.use_bias)
#         # conv = BayesianLinear(input_dim, output_dim, bias = config.use_bias, 
#         #     prior_mean_bias=config.prior_mean_bias, prior_variance_bias=config.prior_variance_bias,
#         #     prior_mean_weight=config.prior_mean_weight, prior_variance_weight=config.prior_variance_weight,)
#         conv = nn.Linear(input_dim, output_dim, bias=config.use_bias)
#         return conv


class APPNP(GCNLinearClassification):
    """ Approximate Page Rank diffusion after MLP feature transformation with a linear classification layer. """

    def __init__(self, input_dim, num_classes, cfg: ModelConfiguration,):
        super().__init__(input_dim, num_classes, cfg)
        self.diffusion = APPNPConv(cfg.appnp.diffusion_iterations, cfg.appnp.teleportation_probability, cached=cfg.cached)

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        super().clear_and_disable_cache()
        self.diffusion.clear_and_disable_cache()
        self.head.clear_and_disable_cache()

    def forward(self, data, sample=None) -> Prediction:
        x, edge_index, edge_weight, sample = self._unpack_inputs_and_sample(data, sample=sample)
        embeddings = [x]
        for num, layer in enumerate(self.convs):
            x = layer(x, edge_index, edge_weight=edge_weight, sample=sample)
            embeddings.append(x)

        # Diffusion
        diffused = self.diffusion(x, edge_index)
        embeddings.append(diffused)
        logits = self.head(diffused, edge_weight=edge_weight, sample=sample)
        embeddings.append(logits)

        return Prediction(
            features = embeddings,
            **{
                INPUTS : data.x,
                LOGITS : logits,
                SOFT_PREDICTIONS : F.softmax(logits, dim=1),
            }
        )

    @staticmethod
    def _make_conv(input_dim, output_dim, config: ModelConfiguration, *args, 
            use_spectral_norm: bool = False, 
            use_bjorck_norm: bool=False, 
            use_forbenius_norm: bool=False, 
            use_rescaling: bool=False, **kwargs):
        """ Method to create a convolution operator with spectral normalization """
        return LinearWithParametrization(input_dim, output_dim, config, *args, **kwargs, use_spectral_norm=use_spectral_norm, use_bjorck_norm=use_bjorck_norm,
            use_rescaling=use_rescaling, use_forbenius_norm=use_forbenius_norm)


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
    elif cfg.model_type == mconst.GAT:
        return GAT(input_dim, output_dim, cfg)
    elif cfg.model_type == mconst.BGCN:
        return BayesianGCN(input_dim, output_dim, cfg)
    elif cfg.model_type == mconst.APPNP:
        return APPNP(input_dim, output_dim, cfg)
    elif cfg.model_type == mconst.GCN_LINEAR_CLASSIFICATION:
        return GCNLinearClassification(input_dim, output_dim, cfg)
    elif cfg.model_type == mconst.GCN_LAPLACE:
        return GCNLinearClassificationLaplaceBackbone(input_dim, output_dim, cfg)
    elif cfg.model_type == mconst.MLP:
        return MLP(input_dim, output_dim, cfg)
    else:
        raise RuntimeError(f'Unsupported model type {cfg.model_type}')
        
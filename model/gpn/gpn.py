""" Implementation taken from https://github.com/stadlmax/Graph-Posterior-Network """

from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.utils as tu
from torch_geometric.data import Data

from .nn import uce_loss, entropy_reg
from .util import apply_mask
from .appnp import APPNPPropagation
from .evidence import Density, Evidence

from configuration import ModelConfiguration
from model.gnn import make_convolutions, make_activation_by_configuration
from model.prediction import *

class GPN(nn.Module):
    """Graph Posterior Network model"""

    @staticmethod
    def _make_linear(input_dim, output_dim, config: ModelConfiguration, *args, **kwargs):
        """ Method to create a convolution operator with spectral normalization """
        return nn.Linear(input_dim, output_dim, bias=config.use_bias)

    def __init__(self, input_dim: int, output_dim: int, cfg: ModelConfiguration):
        super().__init__()

        self.use_batched_flow = cfg.use_batched_flow
        self.alpha_evidence_scale = cfg.alpha_evidence_scale
        self.num_classes = output_dim
        self.latent_size = cfg.latent_size

        dims = [input_dim] + list(cfg.hidden_sizes)
        self.convs = nn.ModuleList([
            nn.Linear(d1, d2, bias=cfg.use_bias)
        ] for d1, d2 in zip(dims[:-1], dims[1:]))

        self.convs = make_convolutions(input_dim, output_dim, cfg, self._make_linear)
        self.act = make_activation_by_configuration(cfg)

        self.latent_encoder = nn.Linear(cfg.hidden_sizes[-1], cfg.latent_size)


        use_batched = True if self.use_batched_flow else False 
        self.flow = Density(
            dim_latent=cfg.latent_size,
            num_mixture_elements=output_dim,
            radial_layers=cfg.num_radial,
            maf_layer=cfg.num_maf,
            gaussian_layers=cfg.num_gaussians,
            use_batched_flow=use_batched)

        self.evidence = Evidence(scale=self.alpha_evidence_scale)

        self.propagation = APPNPPropagation(
            K=cfg.diffusion_iterations,
            alpha=cfg.teleportation_probability,
            add_self_loops=False,
            cached=False,
            normalization='sym')
    
    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        self.propagation.clear_and_disable_cache()

    def get_output_weights(self):
        """ Gets the weights of the output layer. """
        return self.latent_encoder.weight

    def forward(self, data: Data):
        return self.forward_impl(data)

    def forward_impl(self, data: Data) :
        edge_index = data.edge_index if data.edge_index is not None else data.adj_t
        x = data.x

        embeddings = [x]
        for conv in self.convs:
            x = self.act(conv(x))
            embeddings.append(x)

        z = self.latent_encoder(x)
        embeddings.append(z)

        # compute feature evidence (with Normalizing Flows)
        # log p(z, c) = log p(z | c) p(c)
        p_c = self.get_class_probalities(data)
        log_q_ft_per_class = self.flow(z) + p_c.view(1, -1).log()

        if '-plus-classes' in self.alpha_evidence_scale:
            further_scale = self.num_classes
        else:
            further_scale = 1.0

        beta_ft = self.evidence(
            log_q_ft_per_class, dim=self.latent_size,
            further_scale=further_scale).exp()

        alpha_features = 1.0 + beta_ft

        beta = self.propagation(beta_ft, edge_index)
        alpha = 1.0 + beta

        soft = alpha / alpha.sum(-1, keepdim=True)
        log_soft = soft.log()

        max_soft, hard = soft.max(dim=-1)

        return Prediction(
            embeddings,
            inputs = data.x,
            soft = soft,
            epistemic_confidence = alpha[torch.arange(hard.size(0)), hard],
        )

    def get_optimizer(self, lr: float, weight_decay: float) -> Tuple[optim.Adam, optim.Adam]:
        flow_lr = lr if self.params.factor_flow_lr is None else self.params.factor_flow_lr * lr
        flow_weight_decay = weight_decay if self.params.flow_weight_decay is None else self.params.flow_weight_decay

        flow_params = list(self.flow.named_parameters())
        flow_param_names = [f'flow.{p[0]}' for p in flow_params]
        flow_param_weights = [p[1] for p in flow_params]

        all_params = list(self.named_parameters())
        params = [p[1] for p in all_params if p[0] not in flow_param_names]

        # all params except for flow
        flow_optimizer = optim.Adam(flow_param_weights, lr=flow_lr, weight_decay=flow_weight_decay)
        model_optimizer = optim.Adam(
            [{'params': flow_param_weights, 'lr': flow_lr, 'weight_decay': flow_weight_decay},
             {'params': params}],
            lr=lr, weight_decay=weight_decay)

        return model_optimizer, flow_optimizer

    def get_warmup_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        model_optimizer, flow_optimizer = self.get_optimizer(lr, weight_decay)

        if self.params.pre_train_mode == 'encoder':
            warmup_optimizer = model_optimizer
        else:
            warmup_optimizer = flow_optimizer

        return warmup_optimizer

    def get_finetune_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        # similar to warmup
        return self.get_warmup_optimizer(lr, weight_decay)

    def uce_loss(self, prediction: Prediction, data: Data, approximate=True) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_train, y = apply_mask(data, prediction.alpha, split='train')
        reg = self.params.entropy_reg
        return uce_loss(alpha_train, y, reduction='sum'), \
            entropy_reg(alpha_train, reg, approximate=approximate, reduction='sum')

    def loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        uce, reg = self.uce_loss(prediction, data)
        n_train = data.train_mask.sum() if self.params.loss_reduction == 'mean' else 1
        return {'UCE': uce / n_train, 'REG': reg / n_train}

    def warmup_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        if self.params.pre_train_mode == 'encoder':
            return self.CE_loss(prediction, data)

        return self.loss(prediction, data)

    def finetune_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        return self.warmup_loss(prediction, data)

    def likelihood(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_class_probalities(self, data: Data) -> torch.Tensor:
        l_c = torch.zeros(self.params.num_classes, device=data.x.device)
        y_train = data.y[data.train_mask]

        # calculate class_counts L(c)
        for c in range(self.params.num_classes):
            class_count = (y_train == c).int().sum()
            l_c[c] = class_count

        L = l_c.sum()
        p_c = l_c / L

        return p_c

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

from configuration import ModelConfiguration
from model.nn import GCNConv, make_activation_by_configuration
from model.gnn import ModelBase

class FeatureReconstruction(ModelBase):
    """ Reconstruction head for feature reconstruction embedding space to prevent feature collapse. """

    def __init__(self, num_inputs: int, config: ModelConfiguration):
        super().__init__(config)
        self.cached = config.cached
        self._cache = None
        if config.feature_reconstruction.mirror_encoder:
            dims = list(reversed(config.hidden_sizes)) + [num_inputs]
            self.convs = nn.ModuleList()
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                self.convs.append(GCNConv(in_dim, out_dim, add_self_loops=False, cached=config.cached, bias=config.use_bias))
            self.act = make_activation_by_configuration(config)
            self.activation_on_last_layer = config.feature_reconstruction.activation_on_last_layer
        else:
            raise NotImplemented

        self._loss = config.feature_reconstruction.loss
        self._num_samples = config.feature_reconstruction.num_samples
        self.rng = np.random.RandomState(seed=config.feature_reconstruction.seed)

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. Useful after training. """
        for conv in self.convs:
            conv.clear_and_disable_cache()
        self._cache = None
        self.cached = False
    
    def forward(self, data, x: torch.Tensor, sample=None):
        _, edge_index, edge_weight, sample = self._unpack_inputs_and_sample(data, sample=sample)
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if idx < len(self.convs) - 1 or self.activation_on_last_layer:
                x = self.act(x)
        return x

    def loss(self, reconstructed: torch.Tensor, input_features: torch.Tensor, calculate_metrics: bool=True) -> Tuple[torch.Tensor, Dict[str, int]]:
        """ Gets the reconstruction loss. 
        
        Parameteres:
        ------------
        reconstructed : Tensor, shape [*]
            The reconstructed features.
        input_features : Tensor, shape [*]
            The target input features.
        calculate_metrics : bool, optional, default: True
            If additional metrics will be calculated. Note that this may be costly and shouldnt be done every batch.
        
        Returns:
        --------
        loss : Tensor, shape [*]
            The (unaveraged) loss.
        metrics : dict
            Additional metrics like AUROC.
        """
        if self._num_samples > 0:
            sampled_idxs = self.rng.choice(reconstructed.size(0), size=self._num_samples, replace=False)
            reconstructed = reconstructed[sampled_idxs]
            input_features = input_features[sampled_idxs]

        if self._loss == 'l2':
            return F.mse_loss(reconstructed, input_features, reduction='none'), {}
        elif self._loss == 'l1':
            return  F.l1_loss(reconstructed, input_features, reduction='none'), {}
        elif self._loss in ('bce', 'weighted_bce'):
            # Binary cross entropy losses
            if self._cache:
                target = self._cache['target']
                target_npy = self._cache['target_npy']
            else:
                target = (input_features > 0).float()
                target_npy = target.detach().cpu().numpy().astype(bool)
                if self.cached:
                    self._cache = {
                        'target' : target,
                        'target_npy' : target_npy,
                    }
            pos_weight = (target.size(0) / target.sum(0)) - 1
            pos_weight[~torch.isfinite(pos_weight)] = 0.0 # Features might not appear even once
            if calculate_metrics:
                metrics = {
                    'auroc' : roc_auc_score(target_npy.flatten(), reconstructed.detach().cpu().numpy().flatten())
                }
            else:
                metrics = {}

            if self._loss != 'weighted_bce':
                pos_weight = None
            return F.binary_cross_entropy_with_logits(reconstructed, target.float(), pos_weight=pos_weight, reduction='none'), metrics
        else:
            raise ValueError(f'Unsupported feature reconstruction loss type {self._loss}')
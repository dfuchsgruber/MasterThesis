import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from metrics import accuracy
from model.gnn import make_model_by_configuration
from model.prediction import Prediction
from model.reconstruction import ReconstructionLoss
import model.constants as mconst
from torch_geometric.utils import remove_self_loops, add_self_loops
from configuration import ModelConfiguration
import logging
from sklearn.metrics import roc_auc_score

def log_metrics(module: pl.LightningModule, metrics, prefix=None):
    if prefix is None:
        prefix = []
    elif isinstance(prefix, str):
        prefix = [prefix]
    else:
        prefix = list(prefix)
    for metric, value in metrics.items():
        module.log('_'.join(prefix + [metric]), value)

class SemiSupervisedNodeClassification(pl.LightningModule):
    """ Wrapper for networks that perform semi supervised node classification. """
    
    def __init__(self, backbone_configuration: ModelConfiguration, num_input_features, num_classes, learning_rate=1e-2, weight_decay=0.0,):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_configuration"])
        self.backbone = make_model_by_configuration(backbone_configuration, num_input_features, num_classes)
        self.learning_rate = learning_rate
        self.self_loop_fill_value = backbone_configuration.self_loop_fill_value
        self.weight_decay = weight_decay
        self._self_training = False
        self.reconstruction_loss_weight = backbone_configuration.reconstruction.loss_weight
        if self.reconstruction_loss_weight > 0:
            self.reconstruction_loss = ReconstructionLoss(backbone_configuration.reconstruction)

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        self.backbone.clear_and_disable_cache()
        if self.reconstruction_loss_weight > 0:
            self.reconstruction_loss.clear_and_disable_cache()

    @property
    def self_training(self):
        return self._self_training

    @property
    def training_type(self):
        return self.backbone.training_type

    @self_training.setter
    def self_training(self, value: bool):
        if value != self.self_training:
            logging.info(f'Self-training in model changed to {value}')
            self._self_training = value

    def forward(self, batch, *args, remove_edges: bool=False, sample: bool=None, **kwargs) -> Prediction:

        edge_index, edge_weight = batch.edge_index, batch.edge_weight
        
        if remove_edges:
            edge_index = torch.tensor([]).view(2, 0).long().to(edge_index.device)
            edge_weight = torch.tensor([]).view(0).float().to(edge_weight.device)

        # Replace / add self loops with a given fill value
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value = self.self_loop_fill_value, num_nodes=batch.x.size(0))

        batch.edge_index = edge_index
        batch.edge_weight = edge_weight

        return self.backbone(batch, *args, sample=sample, **kwargs)

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def step(self, batch, batch_idx, is_training: bool=True):
        metrics = {}
        output = self(batch)
        logits = output.get_logits(average=True)
        loss = self.cross_entropy_loss(logits[batch.mask], batch.y[batch.mask])
        metrics['cross_entropy'] = loss
        metrics['accuracy'] =  accuracy(logits[batch.mask], batch.y[batch.mask])

        # Self-training
        if self.self_training and is_training:
            with torch.no_grad():
                pred = logits.argmax(1)
            self_training_loss = self.cross_entropy_loss(logits[~batch.mask], pred[~batch.mask])
            metrics['self_training_cross_entropy'] =  self_training_loss
            loss += self_training_loss

        # Autoencoder
        if self.reconstruction_loss_weight > 0:
            reco_loss, reco_proxy, reco_target = self.reconstruction_loss(output.get_features(-2, average=True), batch.edge_index)
            metrics['reconstruction_loss'] =  reco_loss * self.reconstruction_loss_weight
            metrics['reconstruction_auroc'] = roc_auc_score(reco_target.detach().cpu().numpy().astype(bool), reco_proxy.detach().cpu().numpy())
            loss += reco_loss

        # Additional model losses (such as Bayesian KL divergence)
        for name, value in self.backbone.losses(output).items():
            metrics[name] = value
            loss += value
            
        metrics['loss'] = loss
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.step(batch, batch_idx, is_training=True)
        log_metrics(self, metrics, prefix='train')
        return metrics['loss']


    def validation_step(self, batch, batch_idx):
        metrics = self.step(batch, batch_idx, is_training=False)
        log_metrics(self, metrics, prefix='val')
        return metrics['loss']
        

    def get_output_weights(self) -> torch.Tensor:
        """ Gets the weights of the output layer. """
        return self.backbone.get_output_weights()



class Ensemble(pl.LightningModule):
    """ Wrapper class for a model ensemble.
    
    Parameteres:
    ------------
    members : list
        List of torch modules that output predictions.
    num_samples : int
        How many samples to draw from each member.
    sample_at_eval : bool, default: False
        If samples should be drawn by default at evaluation time. Setting it to `True` will enable dropout etc.
        Can be overriden by passing `sample = None` to forward pass.
    """

    def __init__(self, members, num_samples=1, sample_at_eval=False):
        super().__init__()
        self.num_samples = num_samples
        self.members = nn.ModuleList(list(members))
        self.sample_at_eval = sample_at_eval

    def forward(self, *args, sample: bool=None, **kwargs):
        if self.training:
            raise RuntimeError(f'Ensemble is in training mode. Members should be trained separately and switched to eval mode.')
        if sample is None:
            sample = self.sample_at_eval # Fallback
        if sample:
            num_samples = self.num_samples
        else:
            num_samples = 1

        all_predictions = []
        for member in self.members:
            for _ in range(num_samples):
                all_predictions.append(member(*args, sample=sample, **kwargs))
        return Prediction.collate(all_predictions)

    def configure_optimizers(self):  
        raise RuntimeError(f'Ensemble members should be trained by themselves.')

    def training_step(self, batch, batch_idx):
        raise RuntimeError(f'Ensemble members should be trained by themselves.')

    def get_output_weights(self) -> torch.Tensor:
        """ Gets the weights of the output layer for 1-member ensembles. """
        if len(self.members) > 1:
            raise RuntimeError(f'Cant get output weights for a model with multiple ensemble members')
        return self.members[0].get_output_weights() 
  
    def validation_step(self, batch, batch_idx):
        for idx, member in enumerate(self.members):
            metrics = member.step(batch, batch_idx, is_training=False)
            log_metrics(self, metrics, prefix=f'val_member_{idx}')
        logits = self(batch).get_logits(average=True)
        self.log('ensemble_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))
    
    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        for member in self.members:
            member.clear_and_disable_cache()
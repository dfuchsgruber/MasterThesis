import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from metrics import accuracy
from model.gnn import make_model_by_configuration
from model.prediction import Prediction
from torch_geometric.utils import remove_self_loops, add_self_loops
from configuration import ModelConfiguration
import logging

class SemiSupervisedNodeClassification(pl.LightningModule):
    """ Wrapper for networks that perform semi supervised node classification. """

    def __init__(self, backbone_configuration: ModelConfiguration, num_input_features, num_classes, learning_rate=1e-2, weight_decay=0.0,):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = make_model_by_configuration(backbone_configuration, num_input_features, num_classes)
        self.learning_rate = learning_rate
        self.self_loop_fill_value = backbone_configuration.self_loop_fill_value
        self.weight_decay = weight_decay
        self._self_training = False

    @property
    def self_training(self):
        return self._self_training

    @self_training.setter
    def self_training(self, value: bool):
        if value != self.self_training:
            logging.info(f'Self-training in model changed to {value}')
            self._self_training = value

    def forward(self, batch, *args, remove_edges=False, **kwargs):

        edge_index, edge_weight = batch.edge_index, batch.edge_weight
        
        if remove_edges:
            edge_index = torch.tensor([]).view(2, 0).long().to(edge_index.device)
            edge_weight = torch.tensor([]).view(0).float().to(edge_weight.device)

        # Replace / add self loops with a given fill value
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value = self.self_loop_fill_value, num_nodes=batch.x.size(0))

        batch.edge_index = edge_index
        batch.edge_weight = edge_weight

        return Prediction(self.backbone(batch, *args, **kwargs))

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self(batch).get_logits(average=True)
        loss = self.cross_entropy_loss(logits[batch.mask], batch.y[batch.mask])
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))
        if self.self_training:
            with torch.no_grad():
                pred = logits.argmax(1)
            self_training_loss = self.cross_entropy_loss(logits[~batch.mask], pred[~batch.mask])
            self.log('self_training_loss', self_training_loss)
            loss += self_training_loss
        return loss
  
    def validation_step(self, batch, batch_idx):
        logits = self(batch).get_logits(average=True)
        loss = self.cross_entropy_loss(logits[batch.mask], batch.y[batch.mask])
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))

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
    sample_during_training : bool
        If multiple samples will be drawn and averaged even during training (also averages gradients). Defaults to False.
    """

    def __init__(self, members, num_samples=1, sample_during_training=False):
        super().__init__()
        self.num_samples = num_samples
        self.members = nn.ModuleList(list(members))
        self.sample_during_training = sample_during_training

    def forward(self, *args, **kwargs):
        if self.training and not self.sample_during_training:
            num_samples = 1 # Don't sample during training
        else:
            num_samples = self.num_samples
        return Prediction.collate([Prediction.collate(member(*args, **kwargs) for member in self.members) for _ in range(num_samples)])

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
        logits = self(batch).get_logits(average=True)
        loss = F.cross_entropy(logits[batch.mask], batch.y[batch.mask])
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))


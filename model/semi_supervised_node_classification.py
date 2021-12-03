import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from metrics import accuracy
from model.gnn import make_model_by_configuration
from model.prediction import Prediction

class SemiSupervisedNodeClassification(pl.LightningModule):
    """ Wrapper for networks that perform semi supervised node classification. """

    def __init__(self, backbone_configuration, num_input_features, num_classes, learning_rate=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = make_model_by_configuration(backbone_configuration, num_input_features, num_classes)
        self.learning_rate = learning_rate

    def forward(self, batch, *args, remove_edges=False, **kwargs):
        if remove_edges:
            batch.edge_index = torch.tensor([]).view(2, 0).long().to(batch.edge_index.device)
        return Prediction(self.backbone(batch, *args, **kwargs))

    def configure_optimizers(self):  
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self(batch).get_logits(average=True)
        loss = self.cross_entropy_loss(logits[batch.mask], batch.y[batch.mask])
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))
        return loss
  
    def validation_step(self, batch, batch_idx):
        logits = self(batch).get_logits(average=True)
        loss = self.cross_entropy_loss(logits[batch.mask], batch.y[batch.mask])
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))


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
  
    def validation_step(self, batch, batch_idx):
        logits = self(batch).get_logits(average=True)
        loss = F.cross_entropy(logits[batch.mask], batch.y[batch.mask])
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from metrics import accuracy
from model.gnn import make_model_by_configuration

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
        return self.backbone(batch, *args, **kwargs)

    def configure_optimizers(self):  
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self(batch)[-1]
        loss = self.cross_entropy_loss(logits[batch.mask], batch.y[batch.mask])
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))
        return loss
  
    def validation_step(self, batch, batch_idx):
        logits = self(batch)[-1]
        loss = self.cross_entropy_loss(logits[batch.mask], batch.y[batch.mask])
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))
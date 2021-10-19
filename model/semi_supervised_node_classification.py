import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from metrics import accuracy

class SemiSupervisedNodeClassification(pl.LightningModule):
    """ Wrapper for networks that perform semi supervised node classification. """

    def __init__(self, backbone, learning_rate=1e-2):
        super().__init__()
        self.learning_rate = learning_rate
        self.backbone = backbone

    def forward(self, batch):
        return self.backbone(batch)

    def configure_optimizers(self):  
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)[-1]
        loss = self.cross_entropy_loss(logits[batch.mask], batch.y[batch.mask])
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))
        return loss
  
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)[-1]
        loss = self.cross_entropy_loss(logits[batch.mask], batch.y[batch.mask])
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy(logits[batch.mask], batch.y[batch.mask]))

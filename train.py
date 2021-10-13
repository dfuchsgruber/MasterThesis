import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
from metrics import accuracy


def train_model_semi_supervised_node_classification(model, data, mask_train_idx, mask_val_idx, epochs=5, early_stopping_patience=10):
    """ Trains the model for semi-supervised node classification on a graph.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train.
    data : torch_geometric.data.Data
        The graph to train on.
    mask_train_idx : ndarray, shape [N]
        Idxs that are used for training.
    mask_val_idx : ndarray, shape [N]
        Idxs that are used for validation.
    epochs : int
        For how many epochs to train.
        
    Returns:
    --------
    metrics_train : dict
        A dictionary containing a history for each metric on the training set.
    metrics_val : dict
        A dictionary containing a history for each metric on the validation set.
    """

    x, edge_index, y = data.x.float(), data.edge_index.long(), data.y.long()
    if torch.cuda.is_available():
        x, edge_index, y = x.cuda(), edge_index.cuda(), y.cuda()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    metrics_train, metrics_val = defaultdict(list), defaultdict(list)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        y_pred = model(x, edge_index)[-1]
        loss = criterion(y_pred[mask_train_idx], y[mask_train_idx])
        loss.backward()
        optimizer.step()

        metrics_train['loss'].append(loss.item())
        metrics_train['accuracy'].append(accuracy(y_pred[mask_train_idx], y[mask_train_idx]))

        # Validation
        with torch.no_grad():
            val_loss = criterion(y_pred[mask_val_idx], y[mask_val_idx])
            metrics_val['loss'].append(val_loss.item())
            metrics_val['accuracy'].append(accuracy(y_pred[mask_val_idx], y[mask_val_idx]))

            # TODO early stopping

    return dict(metrics_train), dict(metrics_val)


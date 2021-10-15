import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
from metrics import accuracy

class EarlyStopping:
    """ Mintor for early stopping. Training is stopped if a quantity doesn't improve over the average of the last `patience` steps. """

    def __init__(self, patience, modes, improvement_threshold=1e-3, ):
        """ Initializes the early stopping monitor. 
        
        Parameters:
        -----------
        patience : int
            If a quantity doesn't improve over the last `patience` steps, training is stopped.
        modes : dict
            A dict from each metric to monitor to either 'min' or 'max'.
            If 'max', a higher quantity is better. If 'min', a lower quantity is better.
        improvement_threshold : float
            Improvements smaller than this threshold will be disregarded.
        """
        self.patience = patience
        self.modes = modes
        self.improvement_threshold = improvement_threshold

        self.best = {metric : -np.inf for metric in modes}
        self.steps = 0

        self._should_stop = False
        self.best_model_state_dict = None
        self._epoch = 0
        self.best_epoch = None

    def log(self, metrics, model):
        """ Logs a certain metric.
        
        Parameters:
        -----------
        metrics : dict
            The metrics to log.
        model : torch.nn.Module
            The model, so that the current best state dict is also logged.
            
        """
        improvement = False
        for metric in self.modes.keys():
            if self.modes[metric] == 'min':
                value = -metrics[metric]
            elif self.modes[metric] == 'max':
                value = metrics[metric]
            else:
                raise RuntimeError(f'Unsupported mode {self.modes[metric]} for metric {metric}')
            if value >= self.best[metric] + self.improvement_threshold: # Improvement
                improvement = True
                self.best[metric] = value
        if improvement:
            self.steps = 0
            self.best_model_state_dict = model.state_dict()
            self.best_epoch = self._epoch
        else:
            self.steps += 1
            if self.steps >= self.patience:
                self._should_stop = True
        self._epoch += 1


    def should_stop(self):
        """ If training should be stopped. Needs at least `patience + 1` logging steps.
        
        Returns:
        --------
        If training should be stopped.
        """
        return self._should_stop

def train_model_semi_supervised_node_classification(model, data, mask_train_idx, mask_val_idx, epochs=5, 
        early_stopping_patience=10, early_stopping_metrics={'val_loss' : 'min', }, learning_rate=1e-3,):
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
    early_stopping_patience : int
        How many epochs to wait for improvements.
    early_stopping_metrics : dict
        Mapping from metrics to minitor to either 'max' (higher metric is better) or 'min' (lower metric is better).
    learning_rate : float
        Learning rate of the ADAM optimizer.

    Returns:
    --------
    metrics_train : dict
        A dictionary containing a history for each metric on the training set.
    metrics_val : dict
        A dictionary containing a history for each metric on the validation set.
    best_model : dict
        The state dict of the best model.
    best_epoch : int
        Which epoch the model state dict corresponds to.
    """

    x, edge_index, y = data.x.float(), data.edge_index.long(), data.y.long()
    if torch.cuda.is_available():
        x, edge_index, y = x.cuda(), edge_index.cuda(), y.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    monitor = EarlyStopping(early_stopping_patience, early_stopping_metrics)

    metrics_train, metrics_val = defaultdict(list), defaultdict(list)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        y_pred = model(x, edge_index)[-1]
        loss = criterion(y_pred[mask_train_idx], y[mask_train_idx])
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_accuracy = accuracy(y_pred[mask_train_idx], y[mask_train_idx])
        metrics_train['loss'].append(train_loss)
        metrics_train['accuracy'].append(train_accuracy)

        # Validation
        with torch.no_grad():
            val_loss = criterion(y_pred[mask_val_idx], y[mask_val_idx]).item()
            val_accuracy = accuracy(y_pred[mask_val_idx], y[mask_val_idx])
            print(f'Epoch {epoch}: Train loss {train_loss:.4f} Val loss {val_loss:.4f} Train acc {train_accuracy:.4f} Val acc: {val_accuracy:.4f}')
            metrics_val['loss'].append(val_loss)
            metrics_val['accuracy'].append(val_accuracy)
            monitor.log({'val_loss' : val_loss, 'val_accuracy' : val_accuracy}, model)
            if monitor.should_stop():
                print(f'Early stopping criterion reached after {epoch + 1} epochs.')
                break

    return dict(metrics_train), dict(metrics_val), monitor.best_model_state_dict, monitor.best_epoch


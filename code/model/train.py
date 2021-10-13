import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict

def train_model_semi_supervised_node_classification(model, data, train_idx, val_idx, epochs=500, early_stopping_patience=10):
    """ Trains the model for semi-supervised node classification on a graph.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train.
    data : torch_geometric.data.Data
        The graph to train on.
    train_idx : ndarray, shape [n_train]
        Idxs that are used for training.
    val_idx : ndarray, shape [n_val]
        Idxs that are used for validation.
    epochs : int
        For how many epochs to train.
        
    Returns:
    --------
    loss_history : np.array, shape [num_epochs]
        Average validation loss after each epoch.
    accuarcy_history : np.array, shape [num_epochs]
        Validation accuracy after each epoch.
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

        y_pred = model(x, edge_index)
        loss = criterion(y_pred[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        metrics_train['loss'].append()

        # Validation
        with torch.no_grad():
            val_loss = criterion(y_pred[val_idx], y[val_idx])




        # Training
        print(f'### Epoch {epoch + 1} / {epochs}')
        running_loss, running_accuracy = 0.0, 0.0
        model.train()
        for batch_idx, (x, y) in enumerate(data_loader_train):
            if torch.cuda.is_available():
                x, y = x.float().cuda(), y.cuda()
            optimizer.zero_grad() 
            y_pred = model(x)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            running_accuracy += accuracy(y_pred, y)

            loss.backward()
            optimizer.step()

            #print(f'Batch {batch_idx + 1}. Running loss: {running_loss / (batch_idx + 1):.4f}; Running accuracy {running_accuracy / (batch_idx + 1):.4f}\r', end='\r')
        
        # Validation
        val_loss, val_accuracy = evaluate(model, data_loader_val, criterion)
        print(f'Validation loss {val_loss:.4f}; Validation accuracy {val_accuracy:.4f}')
        loss_history.append(val_loss)
        accuracy_history.append(val_accuracy)

    return np.array(loss_history), np.array(accuracy_history)


import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from util import k_hop_neighbourhood

def make_callback_get_features(layer=-2, mask=True, cpu=True, ensemble_average=True):
    """ Creates callback that gets features from the second to last layer. """
    def callback(data, output):
        result = output.get_features(layer, average=ensemble_average) # N, d, [ensemble_size]
        if mask:
            result = result[data.mask]
        if cpu:
            result = result.cpu()
        return result
    return callback

def make_callback_get_predictions(layer=-1, mask=True, cpu=True, ensemble_average=True):
    """ Creates a callback that gets predicted scores from a model ensemble. """
    def callback(data, output):
        logits = make_callback_get_logits(layer=layer, mask=mask, cpu=cpu, ensemble_average=False)(data, output) # N, classes, ensemble
        scores = F.softmax(logits, dim=1)
        if ensemble_average:
            scores = scores.mean(dim=-1)
        return scores
    return callback

def make_callback_get_logits(layer=-1, mask=True, cpu=True, ensemble_average=False):
    """ Creates a callback that gets logits from a model ensemble. """
    def callback(data, output):
        pred = output.get_logits(average=ensemble_average)
        if mask:
            pred = pred[data.mask]
        if cpu:
            pred = pred.cpu()
        return pred
    return callback

def make_callback_get_data(cpu=True):
    """ Creates a callback that gets the entire data object. """
    def callback(data, output):
        if cpu:
            data = data.cpu()
        return data
    return callback

def make_callback_get_data_features(mask=True, cpu=True):
    """ Creates a callback that gets the feature tensor of the input data. """
    def callback(data, output):
        x = data.x
        if mask:
            x = x[data.mask]
        if cpu:
            x = x.cpu()
        return x
    return callback

def make_callback_get_ground_truth(mask=True, cpu=True):
    """ Creates a callback that gets the ground truth. """
    def callback(data, output):
        y = data.y
        if mask:
            y = y[data.mask]
        if cpu:
            y = y.cpu()
        return y
    return callback

def make_callback_count_neighbours_with_attribute(attribute_getter, k, mask=True, cpu=True):
    """ Makes a callback that counts the neighbours with a certain attribute. 
    
    The result(s) of this callback will have shape [N, 2], where the first column is the fraction of neighbour with that attribute
    and the second column is the total number of neighbours.
    """
    def callback(data, output):
        has_attribute = attribute_getter(data, output)
        n = data.x.size(0)
        edge_index = data.edge_index.cpu().numpy()
        A = sp.coo_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(n, n), dtype=bool)
        k_hop_nbs = k_hop_neighbourhood(A, k).astype(int)
        result = torch.tensor(k_hop_nbs.multiply(has_attribute[None, :]).sum(1)).squeeze().long()
        num_nbs = torch.tensor(k_hop_nbs.sum(1)).squeeze().long()
        result = torch.stack([result, num_nbs], 1)
        if mask:
            result = result[data.mask]
        if cpu:
            result = result.cpu()
        return result
    return callback

def make_callback_get_degree(hops=1, mask=True, cpu=True):
    """ Makes a callback that gets the degree of a node in a given k-hop neighbourhood. """
    def callback(data, output):
        n = data.x.size(0)
        edge_index = data.edge_index.cpu().numpy()
        A = sp.coo_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(n, n), dtype=bool)
        k_hop_nbs = k_hop_neighbourhood(A, hops).astype(int)
        result = torch.tensor(np.array(k_hop_nbs.sum(1))).to(data.x.device).squeeze()
        if mask:
            result = result[data.mask]
        if cpu:
            result = result.cpu()
        return result
    return callback

def make_callback_is_ground_truth_in_labels(labels, mask=True, cpu=True):
    """ Makes a callback that identifies all train labels in the ground truth. """
    labels = set(labels)
    def callback(data, output):
        is_train_label = torch.zeros_like(data.y).bool()
        for label in labels:
            is_train_label[data.y == data.label_to_idx.get(label, -1)] = True
        if mask:
            is_train_label = is_train_label[data.mask]
        if cpu:
            is_train_label = is_train_label.cpu()
        return is_train_label
    return callback

# def make_callback_count_neighbours(k, mask=True, cpu=True):
#     """ Makes a callback counts all neighbours in the k-hop neighbood. """
#     return make_callback_count_neighbours_with_attribute(lambda data, output: np.ones(data.x.size(0), dtype=bool), k, mask=mask, cpu=cpu)


def make_callback_get_perturbation_mask(mask=True, cpu=True):
    """ Makes a callback that gets the perturbation masks in the datasets. """
    def callback(data, output):
        perturbation_mask = data.is_perturbed
        if mask:
            perturbation_mask = perturbation_mask[data.mask]
        if cpu:
            perturbation_mask = perturbation_mask.cpu()
        return perturbation_mask
    return callback


def make_callback_get_mask(mask=True, cpu=True):
    """ Makes a callback that gets the masks in the datasets. """
    def callback(data, output):
        _mask = data.mask
        if mask:
            _mask = _mask[data.mask]
        if cpu:
            _mask = _mask.cpu()
        return _mask
    return callback

def make_callback_get_attribute(attribute_getter, mask=True, cpu=True):
    """ Makes a callback that gets a certain attribute from the data and masks it if it is a Tensor. """
    def callback(data, output):
        value = attribute_getter(data, output)
        if isinstance(value, torch.Tensor):
            if mask:
                value = value[data.mask]
            if cpu:
                value = value.cpu()
        return value
    return callback


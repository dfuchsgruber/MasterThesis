import torch
import torch.nn.functional as F
import numpy as np
from util import get_k_hop_neighbourhood

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
    """ Makes a callback that counts the neighbours with a certain attribute. """
    def callback(data, output):
        has_attribute = attribute_getter(data, output)
        neighbours = get_k_hop_neighbourhood(data.edge_index, k, k_min=k)
        result = torch.tensor([
                has_attribute[np.array(neighbours.get(idx, []))].sum() for idx in range(data.x.size(0))
            ]).long()
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

def make_callback_count_neighbours(k, mask=True, cpu=True):
    """ Makes a callback counts all neighbours in the k-hop neighbood. """
    return make_callback_count_neighbours_with_attribute(lambda data, output: np.ones(data.x.size(0), dtype=bool), k, mask=mask, cpu=cpu)


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
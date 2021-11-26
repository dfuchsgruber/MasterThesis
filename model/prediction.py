import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Prediction:
    """ Summarizes the predictions of a model ensemble. Is also used for single models. """

    @staticmethod
    def aggregate(predictions):
        """ Aggregates several predictions into one. 
        
        Parameters:
        -----------
        predictions : list
            The predictions to aggregate

        Returns:
        --------
        agg : Prediction
            The aggregated prediction.
        """
        agg = Prediction(None)
        for pred in predictions:
            for features in pred.features:
                agg.add(features)
        return agg

    def __init__(self, features):
        self.features = []
        if features is not None:
            self.add(features)

    def add(self, features):
        """ Adds one prediction to the ensemble prediction.
        
        Parameters:
        -----------
        features : list
            Features per layer to add to the prediction.
         """
        self.features.append(features)

    def __add__(self, other):
        p = Prediction(self.features)


    def get_features(self, layer, average=True):
        """ Get features of this prediction at a given layer. 
        
        Parameters:
        -----------
        layer : int
            From which layer to extract features.
        average : bool
            If given, the features of all members are averaged.
            (In case of a single member in the ensemble, this will simply get that prediction.)

        Returns:
        --------
        features : torch.Tensor, shape [N, d_layer, (num_members)]
            The features at this layer.
        """
        features = torch.stack([features[layer] for features in self.features], dim=-1)
        if average:
            features = features.mean(-1)
        return features

    def get_logits(self, average=True):
        return self.get_features(-1, average=average)


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import logging

from util import all_equal

from typing import List, Optional, Dict, Any
from torch import Tensor

# --- Normalizations for scores

# Optional attributes that override standard procedures for getting features, scores, logits, etc.
SOFT_PREDICTIONS = 'soft_predictions'
HARD_PREDICTIONS = 'hard_predictions'
LOGITS = 'logits'
INPUTS = 'inputs'

class Prediction:
    """ Summarizes the predictions of a model ensemble. Is also used for single models. 
    
    Parameters:
    -----------
    features : List[Tensor]
        Features from all layers.
        By convention, features[0] should be input features.
        By convention, features[-2] should be the most high level hidden embeddings.
        By convention, features[-1] should be logit-like objects.
        If these conventions are not met, pass attributes SOFT_PREDICTIONS, HARD_PREDICTIONS or LOGITS
    """

    @staticmethod
    def collate(predictions: List[Any]) -> Any:
        """ Collates several predictions into one. 
        
        Parameters:
        -----------
        predictions : List[Prediction]
            The predictions to collate

        Returns:
        --------
        agg : Prediction
            The collated prediction.
        """
        agg = Prediction(None)

        if not all_equal([len(p.features) for p in predictions]):
            raise RuntimeError(f'Trying to aggregate predictions with differing feature sizes.')
        if not all_equal([set(p.attributes.keys()) for p in predictions]):
            raise RuntimeError(f'Trying to aggregate predictions with differing attributes.')

        for p in predictions:
            agg.features += p.features
            for k, v in p.attributes.items():
                agg.attributes[k] += v
        return agg

    def __init__(self, features: Optional[List[Tensor]]=None, **kwargs):
        self.clear()
        if features:
            self.features.append(features)
        self.attributes = defaultdict(list)
        for k, v in kwargs.items():
            self.attributes[k].append(v)
        
    def clear(self):
        """ Clears the predictions. """
        self.features = []
        self.attributes = defaultdict(list)

    def get_inputs(self, average: bool = True) -> Tensor:
        """ Gets inputs stored in the prediction. 
        
        Parameters:
        -----------
        average : bool, optional, default: False
            If set, all different inputs are averaged.

        Returns:
        --------
        features : Tensor, shape [N, D, (num_members)]
            Inputs.
        """
        if INPUTS in self.attributes:
            inputs = self.attributes[INPUTS]
        else:
            inputs = [features[0] for features in self.features]
        if average and len(inputs) > 1:
            logging.warn(f'Accessing inputs of a prediction with {len(inputs)} members and averaging. This is probably not wanted...')
        inputs = torch.stack(inputs, dim=-1)
        if average:
            inputs = inputs.mean(-1)
        return inputs


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
    
    def get_logits(self, average: bool = True) -> Tensor:
        """ Gets logits stored in the prediction. 
        
        Parameters:
        -----------
        average : bool, optional, default: False
            If set, all different logits are averaged.

        Returns:
        --------
        logits : Tensor, shape [N, D, (num_members)]
            Logits.
        """
        if LOGITS in self.attributes:
            logits = self.get(LOGITS, average=False)
        else:
            logits = [f[-1] for f in self.features]
            logits = torch.stack(logits, dim=-1)
        if average:
            logits = logits.mean(-1)
        return logits


    def get_predictions(self, soft: bool = True, average: bool = True) -> Tensor:
        """ Gets the predicted class scores. 
        If the prediction has an attribute `SOFT_PREDICTIONS` or `HARD_PREDICTIONS`, it will use this attribute.
        Otherwise, logits are used and the softmax function applied.

        Parameters:
        -----------
        soft : bool, optional, default: True
            If soft predictions should be given.
        average : bool, optional, default: True
            In the case of an ensemble prediction, if it should be averaged.

        Returns:
        --------
        predictions : torch.Tensor, shape [N, C, (num_members)]
        """
        if SOFT_PREDICTIONS in self.attributes:
            softs = self.get(SOFT_PREDICTIONS, average=average)
        else:
            # In case of logit-based soft scores, average the soft scores, not the logits
            softs = self.get_logits(average=False)
            softs = F.softmax(softs, dim=1)
            if average:
                softs = softs.mean(-1)
        
        if HARD_PREDICTIONS in self.attributes:
            hard = self.get(HARD_PREDICTIONS, average=False)
            if average:
                if hard.size(-1) > 1:
                    raise RuntimeError(f'Dont know how to make sense of an ensemble of hard predictions')
                hard = hard[..., -1]
        else:
            hard = torch.argmax(softs, dim=1)

        if soft:
            return softs
        else:
            return hard

    def get(self, attribute, average: bool = True) -> Tensor:
        """ Gets an attribute stored in the prediction. 
        
        Parameters:
        -----------
        average : bool, optional, default: False
            If set, all different attributes are averaged.

        Returns:
        --------
        attr : Tensor, shape [N, D, (num_members)]
            Requested attribute.
        """
        if attribute not in self.attributes:
            raise ValueError(f'Prediction has no attribute {attribute}')
        attr = torch.stack(self.attributes[attribute], dim=-1)
        if average:
            attr = attr.mean(-1)
        return attr
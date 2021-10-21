import torch
import numpy as np
import torch_geometric.transforms as T

class MaskTransform(T.BaseTransform):
    """ Sets a mask to the dataset. """

    def __init__(self, mask):
        super().__init__()
        self.mask = mask
    
    @torch.no_grad()
    def __call__(self, data):
        data.mask = self.mask
        return data

class MaskLabelsTransform(T.BaseTransform):
    """ Sets a mask that only selects certain labels. """
    
    def __init__(self, select_labels):
        super().__init__()
        self.select_labels = select_labels
        
    @torch.no_grad()
    def __call__(self, data):
        mask = torch.zeros_like(data.mask)
        for label in self.select_labels:
            mask[data.y == label] = True
        mask &= data.mask
        data.mask = mask
        return data

class IdentityTransform(T.BaseTransform):
    """ Dummy transformation that does nothing. """

    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def __call__(self, data):
        return data
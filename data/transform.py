import torch
import numpy as np
import torch_geometric.transforms as T

class MaskTransform(T.BaseTransform):
    """ Sets a mask to the dataset. """

    def __init__(self, mask):
        super().__init__()
        self.mask = torch.tensor(mask).bool()
    
    @torch.no_grad()
    def __call__(self, data):
        data.mask = self.mask
        return data

class MaskLabelsTransform(T.BaseTransform):
    """ Sets a mask that only selects certain labels. 
    
    Parameters:
    -----------
    select_labels : iterable
        All labels that the mask in the dataset will select.
    remap_labels : bool
        If True, these selected labels will be remapped to indices (0, 1, ..., k).
        This way no problems occur when training.
    """
    
    def __init__(self, select_labels, remap_labels=True):
        super().__init__()
        self.select_labels = tuple(select_labels)
        self.remap_labels = remap_labels
        
    @torch.no_grad()
    def __call__(self, data):
        mask = torch.zeros_like(data.mask)
        for label in self.select_labels:
            mask[data.y == label] = True
        mask &= data.mask
        data.mask = mask

        if self.remap_labels: # Remap the labels
            for num, label in enumerate(self.select_labels):
                # Swap label `label` and `num`
                idx_num, idx_label = data.y == num, data.y == label
                data.y[idx_num] = label
                data.y[idx_label] = num
                # Change `data.label_to_idx` accordingly
                idx_to_label = {v : k for k, v in data.label_to_idx.items()}
                label_to_idx = data.label_to_idx.copy()
                label_to_idx[idx_to_label[num]] = label
                label_to_idx[idx_to_label[label]] = num
                data.label_to_idx = label_to_idx

        return data

class IdentityTransform(T.BaseTransform):
    """ Dummy transformation that does nothing. """

    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def __call__(self, data):
        return data
import numpy as np
import torch
import os.path as osp
from collections import defaultdict
from warnings import warn
from torch_geometric.data import Dataset, Data
import gust.datasets
from torch_geometric.transforms import Compose
import pytorch_lightning as pl

class SingleGraphDataset(Dataset):
    """ Dataset that only consists of a single graph with vertex masks. 
    Can be used for semi-supervised node classification. 
    
    Parameters:
    -----------
    x : ndarray, shape [N, D]
        Attribute matrix.
    edge_index : ndarray, shape [2, E]
        Edge endpoints.
    y : ndarray, shape [N]
        Labels.
    vertex_to_idx : dict
        Mapping from vertex name to index.
    label_to_idx : dict
        Mapping from label name to index.
    mask : ndarray, shape [N]
        Vertices masked by this dataset.
    transform : torch_geometric.transform.BaseTransform or None
        Transformation to apply to the dataset. If `None` is given, the identity transformation is used.
    is_perturbed : ndarray, shape [N] or None
        If given, a mask to identify veritces with alternated features.
    """

    def __init__(self, x, edge_index, y, vertex_to_idx, label_to_idx, mask, transform=None, is_perturbed=None, feature_to_idx=None, edge_weight=None):

        if transform is None:
            transform = Compose([])
        super().__init__(transform=transform)

        self._data = Data(x=torch.tensor(x).float(), edge_index=torch.tensor(edge_index).long(),
            y=torch.tensor(y).long(), mask=torch.tensor(mask).bool())
        self._data.vertex_to_idx = vertex_to_idx
        self._data.label_to_idx = label_to_idx
        if feature_to_idx is not None:
            self._data.feature_to_idx = feature_to_idx
        self._is_perturbed = is_perturbed
        if edge_weight is not None:
            self._data.edge_weight = edge_weight
        else:
            self._data.edge_weight = torch.ones(self._data.edge_index.size(1))
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        assert idx == 0, f'ŚingleGraphDataset only has a single graph! (Trying to index with {idx}...)'
        data = self._data.clone() # Allows for in-place transformations
        if self._is_perturbed is not None:
            data.is_perturbed = torch.tensor(self._is_perturbed).bool()
        return self.transform(data) # This way we can apply in-place transforms

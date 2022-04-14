import numpy as np
import torch
from warnings import warn
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import Compose
import logging

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
    
    Additional kwargs will be set as attributes to the graph.
    """

    def __init__(self, data, transform=None):

        if transform is None:
            transform = Compose([])
        super().__init__(transform=transform)
        self.data = data
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        assert idx == 0, f'{self.__class__} only has a single element. It cant be indexed by {idx}'
        data = self.data.clone() # Allows for in-place transformations
        return self.transform(data) # This way we can apply in-place transforms

    @staticmethod
    def build(x, edge_index, y, vertex_to_idx, label_to_idx, mask, transform=None, edge_weight=None, is_out_of_distribution = None, **kwargs ):
        """ 
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
        Additional kwargs will be set as attributes to the graph.
        
        Returns:
        --------
        dataset : SingleGraphDataset
            The dataset.
        """
        print(f'Build SingleGraphDataset with additional attribute {kwargs.keys()}')
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).float()
        else:
            edge_weight = torch.tensor(edge_weight)
        if is_out_of_distribution is None:
            is_out_of_distribution = torch.zeros(x.shape[0]).bool()
        else:
            is_out_of_distribution = torch.tensor(is_out_of_distribution)
        attributes = {}
        for attribute, value in kwargs.items():
            if isinstance(value, np.ndarray):
                attributes[attribute] = torch.tensor(value)
            elif isinstance(value, torch.Tensor):
                attributes[attribute] = value.clone()
            else:
                attributes[attribute] = value
        data = Data(
            x = torch.tensor(x).float(),
            edge_index = torch.tensor(edge_index).long(),
            y = torch.tensor(y).long(),
            mask = torch.tensor(mask).bool(),
            vertex_to_idx = vertex_to_idx,
            label_to_idx = label_to_idx,
            edge_weight = edge_weight,
            is_out_of_distribution = is_out_of_distribution,
            **attributes
        )

        return SingleGraphDataset(data, transform=transform)
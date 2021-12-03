import torch
import numpy as np
import torch_geometric.transforms as T
from data.util import graph_select_labels, compress_labels
from torch_geometric.data import Data

class PerturbationTransform(T.BaseTransform):
    """ Perturbs certain vertices with noise. """

    def __init__(self, noise_type='bernoulli', perturb_only_in_mask=True, budget=0.1, **kwargs):
        self.noise_type = noise_type
        self.perturb_only_in_mask = perturb_only_in_mask
        self.budget = budget
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @torch.no_grad()
    def __call__(self, data):
        if self.perturb_only_in_mask:
            idxs = torch.where(data.mask)[0].numpy()
        else:
            idxs = torch.arange(data.x.size(0)).numpy()
        num_to_perturb = int(idxs.shape[0] * self.budget)
        idxs = torch.tensor(np.random.choice(idxs, num_to_perturb, replace=False))
        noise = torch.zeros((num_to_perturb, data.x.size(1)))
        if self.noise_type.lower() in ('bernoulli', ):
            p = getattr(self, 'p', 0.5)
            noise = noise.bernoulli(p)
            noise /= noise.sum(dim=-1).unsqueeze(-1) + 1e-12
        elif self.noise_type.lower() in ('gaussian', 'normal', 'white'):
            scale = getattr(self, 'scale', 1.0)
            noise = noise.normal_()
            noise *= scale
        else:
            raise RuntimeError(f'Unknown perturbation type {self.noise_type}')
        data.x[idxs] = noise
        perturbation_mask = torch.zeros(data.x.size(0)).bool()
        perturbation_mask[idxs] = True
        return data, perturbation_mask

class RemoveEdgesTransform(T.BaseTransform):
    """ Removes all edges in the dataset. """

    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def __call__(self, data):
        data.edge_index = torch.tensor([]).view((2, 0)).long()
        return data

class MaskTransform(T.BaseTransform):
    """ Sets a mask to the dataset. """

    def __init__(self, mask):
        super().__init__()
        self.mask = torch.tensor(mask).bool()
    
    @torch.no_grad()
    def __call__(self, data):
        data.mask = self.mask
        return data

class RemoveLabelsTransform(T.BaseTransform):
    """ Removes vertices from a graph (doesn't mask them!) that have certain labels. Also removes all edges with 
    an endpoint in the set of vertices to remove. 
    
    Parameters:
    -----------
    select_labels : iterable
        Vertices of this label will be kept.
    compress_labels : bool
        If True, labels will be remapped to (0, 1, ...k).
    """

    def __init__(self, select_labels, compress_labels=True):
        self.select_labels = tuple(select_labels)
        self.compress_labels = compress_labels

    @torch.no_grad()
    def __call__(self, data):

        data_mask = data.mask # Original vertex mask that needs to be preserved as well

        x, edge_index, y, vertex_to_idx, label_to_idx, mask = graph_select_labels(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), 
            data.vertex_to_idx, data.label_to_idx, self.select_labels, connected=True, 
            compress_labels=self.compress_labels)
        data_mask = data_mask[mask]

        data.x = torch.tensor(x).float()
        data.y = torch.tensor(y).long()
        data.edge_index = torch.tensor(edge_index).long()
        data.mask = data_mask
        data.vertex_to_idx = vertex_to_idx

        return data

class MaskLabelsTransform(T.BaseTransform):
    """ Sets a mask that only selects certain labels. 
    
    Parameters:
    -----------
    select_labels : iterable
        All labels that the mask in the dataset will select.
    compress_labels : bool
        If True, these selected labels will be remapped to indices (0, 1, ..., k).
        This way no problems occur when training.
    remove_edges : bool
        If True, all vertices (and edges with at least one end point there) will be removed from the graph.
    """
    
    def __init__(self, select_labels, compress_labels=True):
        super().__init__()
        self.select_labels = tuple(select_labels)
        self.compress_labels = compress_labels
        
    @torch.no_grad()
    def __call__(self, data):

        # Refine the mask with selected labels
        mask = torch.zeros_like(data.mask)
        for label in self.select_labels:
            mask[data.y == label] = True
        data.mask &= mask

        if self.compress_labels: 
            y, label_to_idx, compression = compress_labels(data.y.numpy(), data.label_to_idx)
            data.y = torch.tensor(y).long()
            data.label_to_idx = label_to_idx

        return data
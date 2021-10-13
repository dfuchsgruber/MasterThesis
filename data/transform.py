import torch
import numpy as np
import torch_geometric.transforms as T
from util import graph_normalize

class NormalizeGraph(T.BaseTransform):
    """ A callable transformation to a data sample that selects the largest connected component and removes underrepresnted classes. """

    def __init__(self, min_class_count=0.0, verbose=True):
        """ Initializes the transformation.

        Parameters:
        -----------  
        min_class_count : float
            If a portion of vertices of a class is less than this value, it will be removed. Setting this parameter to 0.0 effectively ignores it alltogether.
            Setting it to k / p ensures that if a portion of the data with size p*n is sampled stratifyingly, at least k samples of this class are retained.
        verbose : bool
            If True, normalization steps are printed.
        """
        super().__init__()
        self.min_class_count = min_class_count
        self.verbose = verbose

    def __call__(self, data):
        x, edge_index, y, vertex_to_idx, label_to_idx = graph_normalize(data.x.numpy(), data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, 
            data.label_to_idx, make_symmetric=False, min_class_count=self.min_class_count, verbose=self.verbose)
        data.x = torch.tensor(x)
        data.edge_index = torch.tensor(edge_index)
        data.y = torch.tensor(y)
        data.vertex_to_idx = vertex_to_idx
        data.label_to_idx = label_to_idx
        return data

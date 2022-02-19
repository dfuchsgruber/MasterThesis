# Based on https://github.com/stadlmax/Graph-Posterior-Network/blob/main/gpn/models/gdk.py

import scipy.sparse as sp
import numpy as np
import math
import torch
import torch.nn as nn
import torch_scatter
from torch import Tensor
import networkx as nx
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path_length
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import logging

from configuration import ModelConfiguration



def gaussian_kernel(x: Tensor, sigma: float = 1.0) -> Tensor:
    sigma_scale = 1.0 / (sigma * math.sqrt(2 * math.pi))
    k_dis = torch.exp(-torch.square(x)/ (2 * sigma * sigma))
    return sigma_scale * k_dis


class GDK:
    """ Parameterless Graph Dirichlet Kernel baseline """

    def __init__(self, config: ModelConfiguration):
        self.cached = config.cached
        self._alpha_cache = None
        self.vertices_train = None
        self.num_classes = None

    def forward(self, batch: Data, *args, **kwargs):
        if self.vertices_train is None:
            raise RuntimeError(f'Set the vertices the GDK is fit to first.')
        batch = T.AddSelfLoops()(batch)
        n = batch.x.size(0)

        idx_train = [batch.vertex_to_idx[v] for v in self.vertices_train if v in batch.vertex_to_idx]
        y_train = batch.y[idx_train]
        evidence = torch.zeros((n, self.num_classes), device=batch.y.device)

        edge_index = batch.edge_index.numpy()
        A = sp.coo_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(n, n)).tocsr()
        distances = sp.csgraph.shortest_path(A, indices=idx_train)
        logging.info('GDK - Computed shortest path distances.')
        distances[~np.isfinite(distances)] = 1e10 # large value
        distances = torch.tensor(distances)




# Based on https://github.com/stadlmax/Graph-Posterior-Network/blob/main/gpn/models/gdk.py

from functools import reduce
import scipy.sparse as sp
import numpy as np
import math
import torch
import torch.nn as nn
import torch_scatter
from torch import Tensor
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.data import Data
import logging

from configuration import ModelConfiguration
from model.parameterless import ParameterlessBase
from model.prediction import *

def gaussian_kernel(x: Tensor, sigma: float = 1.0) -> Tensor:
    sigma_scale = 1.0 / (sigma * math.sqrt(2 * math.pi))
    k_dis = torch.exp(-torch.square(x)/ (2 * sigma * sigma))
    return sigma_scale * k_dis


class GraphDirichletKernel(ParameterlessBase):
    """ Parameterless Graph Dirichlet Kernel baseline """

    def __init__(self, config: ModelConfiguration, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = config.gdk.sigma
        self.reduction = config.gdk.reduction
        self.self_loop_fill_value = config.self_loop_fill_value

    def forward(self, batch: Data, *args, **kwargs) -> Prediction:
        idxs_fit = self.get_fit_idxs(batch)

        batch = T.AddSelfLoops(fill_value=self.self_loop_fill_value)(batch)
        n = batch.x.size(0)

        y_train = batch.y[idxs_fit]

        # Evidence is kernelized shorest path distance
        edge_index = batch.edge_index.numpy()
        A = sp.coo_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(n, n)).tocsr()
        distances = sp.csgraph.shortest_path(A, indices=idxs_fit).T # N, num_fit
        logging.info('GDK - Computed shortest path distances.')
        distances[~np.isfinite(distances)] = 1e10 # large value
        distances = gaussian_kernel(torch.tensor(distances), sigma=self.sigma)

        evidence = torch_scatter.scatter(distances, y_train, dim=1, reduce=self.reduction, dim_size=self.num_classes) # N, num_classes
        alpha = 1.0 + evidence

        soft = alpha / (alpha.sum(-1, keepdim=True) + 1e-10)
        hard = alpha.argmax(1)

        return Prediction(
            features=None,
            # The evidence can be used as proxy for uncertainty
            # We consider: i) evidence for the predicted class ii) evidence for all classes
            evidence_prediction = alpha[torch.arange(evidence.size(0)), hard],
            evidence_total = alpha.sum(1),
            evidence_structure_prediction = evidence[torch.arange(evidence.size(0)), hard],
            evidence_structure_total = evidence.sum(1),
            **{
                SOFT_PREDICTIONS : soft,
                HARD_PREDICTIONS : hard,
            }
        )


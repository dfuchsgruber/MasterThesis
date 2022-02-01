from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from configuration import AutoencoderConfiguration


class EdgeSampler:
    """ Samples edges from an adjacency matrix in a balanced way. """


    def __init__(self, seed=1337):
        self.rng = np.random.RandomState(seed=seed)

    def _sample_negative(self, edge_index: torch.Tensor, num: int, num_vertices: int) -> torch.Tensor:
        """ Samples edge indices that are not in `edge_index`. 
        
        Parameters:
        -----------
        edge_index : torch.Tensor, shape [2, N]
            Positive pairs.
        num : int
            How many negative pairs to sample.
        num_vertices : int
            How many vertices there are.
        
        Returns:
        --------
        negatives : torch.Tensor, shape [2, N]
            Negative pairs.
        """

        if num_vertices**2 - edge_index.size(1) < num:
            raise ValueError(f'Cant sample {num} negative links from a {num_vertices}x{num_vertices} adjacency matrix and {edge_index.size(1)} positive links.')

        positives = set((min(i, j), max(i, j)) for i, j in edge_index.T.tolist())
        negatives = set()
        while len(negatives) < num:
            negatives |= set((min(i, j), max(i, j)) for i, j in self.rng.choice(num_vertices, size=(num - len(negatives), 2), replace=True))
            negatives -= positives
        return torch.Tensor(list(negatives)).T.to(edge_index.device)

    def sample(self, edge_index: torch.Tensor, num: int = 100, num_vertices=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Samples positive and negative edges.  
        
        Parameters:
        -----------
        edge_index : torch.Tensor, shape [2, N]
            Positive pairs.
        num : int
            How many negative pairs to sample.
        num_vertices : int
            How many vertices there are.
            
        Returns:
        --------
        pos : torch.Tensor, shape [2, num]
            Positive indices.
        neg : torch.Tensor, shape [2, num]
            Negative indices.
        """

        if num_vertices is None:
            num_vertices = int(edge_index.max().item()) + 1
        idx_pos = self.rng.choice(edge_index.size(1), size=num, replace=False).tolist()
        return edge_index[:, idx_pos], self._sample_negative(edge_index, num, num_vertices)
        

class ReconstructionLoss(nn.Module):
    """ Module that implements graph reconstruction loss by sampling positive and negative examples in each forward call. 
    
    Parameters:
    -----------
    seed : int, default: 1337
        Seed for the sampling procedure.
    num_samples: int, default: 100
        How many samples to draw in each forward pass.
    """

    def __init__(self, config: AutoencoderConfiguration):
        super().__init__()
        self.num_samples = config.num_samples
        self._sample = config.sample
        if self._sample:
            self._sampler = EdgeSampler(seed=config.seed)
        else:
            raise NotImplemented

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor):
        """ Samples targets for the reconstruction loss. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Feature representations
        edge_index : torch.Tensor, shape [2, E]
            Edges for the graph.
        
        Returns:
        --------
        logits : torch.Tensor, shape [K]
            Logits from vertex inner products.
        gnd : torch.Tensor, shape [K]
            Targets. That is, in indicator if an edge is present between each pair.
        """
        if self._sample:
            pos, neg = self._sampler.sample(edge_index, num=self.num_samples, num_vertices=features.size(0))

            gnd = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))], 0).float().to(features.device)
            idx = torch.cat([pos, neg], 1).long()

            z_u = torch.index_select(features, 0, idx[0, :])
            z_v = torch.index_select(features, 0, idx[1, :])
            logits = torch.sum(z_u * z_v, -1)
            return logits, gnd
        else:
            raise NotImplemented
    






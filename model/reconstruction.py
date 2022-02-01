from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

import model.constants as mconst
from configuration import ReconstructionConfiguration


class EdgeSampler:
    """ Stateful anchored edge and non-edge sampling. 
    
    Parameters:
    -----------
    seed : int
        The seed for the state of the sampler.
    cached : bool
        If the adjacency matrix should be cached.
    max_attempts : int
        Maximal number of attempts to find anchor sets and negative endpoints.
    """

    def __init__(self, seed: int=1337, cached: bool=True, max_attempts: int=100):
        self.rng = np.random.RandomState(seed=seed)
        self.cached = cached
        self._adj = None # Cached row-adjacency matrix
        self.max_attempts = max_attempts

    def sample(self, edge_index: torch.Tensor, num: int = 100, num_vertices: Optional[int]=None) -> torch.Tensor:
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
        idx : torch.Tensor, shape [3, num]
            Indices of anchor, positive and negative sample.
        """
        if num_vertices is None:
            num_vertices = int(edge_index.max().item()) + 1
        if self._adj is not None:
            A = self._adj
        else:
            A = sp.coo_matrix((np.ones(edge_index.size(1)), edge_index.detach().cpu().numpy()), shape=(num_vertices, num_vertices))
            A = ((A + A.T + sp.identity(num_vertices)) != 0).tocsr() 
            if self.cached:
                self._adj = A
           
        # Sample positive links (which implies the anchors)
        for _ in range(self.max_attempts):
            idx_pos = self.rng.choice(edge_index.size(1), size=num, replace=False)
            pos = edge_index[:, idx_pos]
            anchors = pos[0, :].detach().cpu().numpy()
            if (A[anchors, :].sum(1) < num_vertices).all():
                break
        else:
            raise RuntimeError(f'Could not sample edges such that all anchors have non-existent links')
        
        # Sample negative endpoints
        neg_endpoints = self.rng.choice(num_vertices, size=len(anchors), replace=True)
        for _ in range(self.max_attempts):
            is_edge = np.array(A[anchors, :].todense()[np.arange(num), neg_endpoints]).flatten()
            if not is_edge.any():
                break
            else:
                neg_endpoints[is_edge] = self.rng.choice(num_vertices, size=int(is_edge.sum()), replace=True)
        else:
            raise RuntimeError(f'Could not sample negative endpoints...')
            
        for u, v in zip(anchors, neg_endpoints):
            if A[u, v]:
                print(anchors, neg_endpoints)
                raise
        
        neg_endpoints = torch.Tensor(neg_endpoints).view((1, -1)).long().to(pos.device)
        return torch.cat((pos, neg_endpoints), 0)

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        self.cached = False
        self._adj = None

class ReconstructionLoss(nn.Module):
    """ Module that implements graph reconstruction loss by sampling positive and negative examples in each forward call. 
    
    Parameters:
    -----------
    seed : int, default: 1337
        Seed for the sampling procedure.
    num_samples: int, default: 100
        How many samples to draw in each forward pass.
    """

    def __init__(self, config: ReconstructionConfiguration):
        super().__init__()
        self.num_samples = config.num_samples
        self.reconstruction_type = config.reconstruction_type
        self.margin_constrastive_loss = config.margin_constrastive_loss
        self._sample = config.sample
        if self._sample:
            self._sampler = EdgeSampler(seed=config.seed, cached=config.cached)
        else:
            raise NotImplemented


    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        if self._sample:
            self._sampler.clear_and_disable_cache()

    def _sample_triplets(self, features: torch.Tensor, edge_index: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Samples triplets from edges with an anchor and selects corresponding features.
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Feature representations
        edge_index : torch.Tensor, shape [2, E]
            Edges for the graph.
        
        Returns:
        --------
        feature_anchor : torch.Tensor, shape [K, D]
            Features of the anchor.
        features_pos : torch.Tensor, shape [K, D]
            Positive features.
        features_neg : torch.Tensor, shape [K, D]
            Negative features.
        """
        idx = self._sampler.sample(edge_index, num=self.num_samples, num_vertices=features.size(0))
        features_anchor = torch.index_select(features, 0, idx[0, :])
        features_pos = torch.index_select(features, 0, idx[1, :])
        features_neg = torch.index_select(features, 0, idx[2, :])
        return features_anchor, features_pos, features_neg

    def forward_energy(self, features: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass with an energy loss that drives vertices on edges together and vertices not on edges apart. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Feature representations
        edge_index : torch.Tensor, shape [2, E]
            Edges for the graph.
        
        Returns:
        --------
        loss : torch.Tensor, shape []
            Loss value.
        proxy : torch.Tensor, shape [K]
            Proxy for edge reconstruction.
        target : torch.Tensor, shape [K]
            Targets. That is, in indicator if an edge is present between each pair.
        """
        if self._sample:
            z_a, z_p, z_n = self._sample_triplets(features, edge_index)
            dist_pos = torch.norm(z_a - z_p, p=2, dim=1)
            dist_neg = torch.norm(z_a - z_n, p=2, dim=1)

            loss = (dist_pos**2 + torch.exp(-dist_neg)).mean()

            # Use distance * (-1) as proxy, as larger values indicate a lower likelihood of edges
            proxy = -torch.cat((dist_pos, dist_neg)) 
            labels = torch.cat((torch.ones_like(dist_pos), torch.zeros_like(dist_neg))).float()
            return loss, proxy, labels
        else:
            raise NotImplemented

    def forward_triplet(self, features: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass with contrastive triplet loss, that drives vertices on edges together and vertices not on edges apart. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Feature representations
        edge_index : torch.Tensor, shape [2, E]
            Edges for the graph.
        
        Returns:
        --------
        loss : torch.Tensor, shape []
            Loss value.
        proxy : torch.Tensor, shape [K]
            Proxy for edge reconstruction.
        target : torch.Tensor, shape [K]
            Targets. That is, in indicator if an edge is present between each pair.
        """
        if self._sample:
            z_a, z_p, z_n = self._sample_triplets(features, edge_index)
            dist_pos = torch.norm(z_a - z_p, p=2, dim=1)
            dist_neg = torch.norm(z_a - z_n, p=2, dim=1)
            loss = torch.clamp(dist_pos - dist_neg + self.margin_constrastive_loss, min=0.0).mean()

            # Use distance * (-1) as proxy, as larger values indicate a lower likelihood of edges
            proxy = -torch.cat((dist_pos, dist_neg)) 
            labels = torch.cat((torch.ones_like(dist_pos), torch.zeros_like(dist_neg))).float()
            return loss, proxy, labels
        else:
            raise NotImplemented

    def forward_autoencoder(self, features: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass with autoencoder loss, i.e. reconstruction of edges with binary cross entropy and logits as inner product. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Feature representations
        edge_index : torch.Tensor, shape [2, E]
            Edges for the graph.
        
        Returns:
        --------
        loss : torch.Tensor, shape []
            Loss value.
        proxy : torch.Tensor, shape [K]
            Proxy for edge reconstruction.
        target : torch.Tensor, shape [K]
            Targets. That is, in indicator if an edge is present between each pair.
        """
        if self._sample:
            z_a, z_p, z_n = self._sample_triplets(features, edge_index)
            logits_p = torch.sum(z_a * z_p, -1)
            logits_n = torch.sum(z_a * z_n, -1)

            proxy = torch.cat((logits_p, logits_n))
            labels = torch.cat((torch.ones_like(logits_p), torch.zeros_like(logits_n))).float()
            return F.binary_cross_entropy_with_logits(proxy, labels, reduction='mean'), proxy, labels
        else:
            raise NotImplemented

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Samples targets for the reconstruction loss. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Feature representations
        edge_index : torch.Tensor, shape [2, E]
            Edges for the graph.
        
        Returns:
        --------
        loss : torch.Tensor, shape []
            Loss value.
        proxy : torch.Tensor, shape [K]
            Proxy for edge reconstruction.
        target : torch.Tensor, shape [K]
            Targets. That is, in indicator if an edge is present between each pair.
        """
        if self.reconstruction_type == mconst.AUTOENCODER:
            return self.forward_autoencoder(features, edge_index)
        elif self.reconstruction_type == mconst.TRIPLET:
            return self.forward_triplet(features, edge_index)
        elif self.reconstruction_type == mconst.ENERGY:
            return self.forward_energy(features, edge_index)
        else:
            raise ValueError(f'Unsupported reconstruction type {self.reconstruction_type}')



if __name__ == '__main__':
    s = EdgeSampler()
    edge_index = torch.Tensor([[0, 1, 1, 2], [1, 2, 3, 3]])
    s.sample(edge_index,  num=3, num_vertices=100)



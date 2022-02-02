""" Implementation taken from https://github.com/stadlmax/Graph-Posterior-Network """

from typing import Optional, Tuple

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, List, Any
import torch
import torch.nn as nn
from torch import Tensor
import torch_geometric.utils as tu
from torch_geometric.typing import Adj
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, mul, sum
import logging

def propagation_wrapper(
        propagation: callable, x: Tensor, edge_index: Adj,
        unc_node_weight: Optional[Tensor] = None,
        unc_edge_weight: Optional[Tensor] = None,
        node_normalization: str = 'none',
        return_normalizer: bool = False,
        **kwargs) -> Tensor:
    """wraps default propagation layer with the option of weighting edges or nodes additionally

    Args:
        propagation (callable): original propagation method
        x (Tensor): node features
        edge_index (Adj): edges
        unc_node_weight (Optional[Tensor], optional): additional weight of nodes. Defaults to None.
        unc_edge_weight (Optional[Tensor], optional): additional weight of edges. Defaults to None.
        node_normalization (str, optional): mode of node normalization ('none', 'reweight', 'reweight_and_scale'). Defaults to 'none'.
        return_normalizer (bool, optional): whether or whether not to return normalization factor. Defaults to False.

    Raises:
        AssertionError: raised if unsupported mode of normalization is passed

    Returns:
        Tensor: node features after propagation
    """

    kwargs.setdefault('edge_weight', None)
    edge_weight = kwargs['edge_weight']

    if unc_node_weight is None:
        unc_node_weight = torch.ones_like(x[:, 0]).view(-1, 1)

    # original scale of weighting
    ones = torch.ones_like(unc_node_weight)

    if unc_edge_weight is None:
        x = torch.cat([unc_node_weight * x, unc_node_weight, ones], dim=-1)
        x = propagation(x, edge_index=edge_index, **kwargs)
        dif_ones = x[:, -1].view(-1, 1)
        dif_w = x[:, -2].view(-1, 1)
        dif_x = x[:, 0:-2]

    # unc_edge_weight is not None
    else:
        unc_edge_weight = unc_edge_weight if edge_weight is None else edge_weight * unc_edge_weight

        # diffuse 1 with previous weighting
        dif_ones = propagation(ones, edge_index=edge_index, **kwargs)

        # diffuse x, w with new weighting
        kwargs['edge_weight'] = unc_edge_weight
        x = torch.cat([unc_node_weight * x, unc_node_weight], dim=-1)
        x = propagation(x, edge_index=edge_index, **kwargs)
        dif_w = x[:, -1].view(-1, 1)
        dif_x = x[:, 0:-1]

    if node_normalization in ('reweight_and_scale', None):
        # sum_u c_vu * (sum_u c_vu * w_u * x_u) / (sum_u c_vu * w_u)
        x = dif_ones * dif_x / dif_w

    elif node_normalization == 'reweight':
        x = dif_x / dif_w

    elif node_normalization == 'none':
        dif_w = None
        x = dif_x

    else:
        raise AssertionError

    if return_normalizer:
        return x, dif_w
    return x


def mat_norm(edge_index: Adj, edge_weight: Optional[Tensor] = None, num_nodes: Optional[int] = None,
             add_self_loops: bool = True, dtype: Optional[Any] = None,
             normalization: str = 'sym', **kwargs) -> Adj:
    """computes normalization of adjanceny matrix

    Args:
        edge_index (Adj): representation of edges in graph
        edge_weight (Optional[Tensor], optional): optional tensor of edge weights. Defaults to None.
        num_nodes (Optional[int], optional): number of nodes. Defaults to None.
        add_self_loops (bool, optional): flag to add self-loops to edges. Defaults to True.
        dtype (Optional[Any], optional): dtype . Defaults to None.
        normalization (str, optional): ['sym', 'gcn', 'in-degree', 'out-degree', 'rw', 'in-degree-sym', 'sym-var']. Defaults to 'sym'.

    Raises:
        AssertionError: raised if unsupported normalization is passed to the function

    Returns:
        Adj: normalized adjacency matrix
    """

    if normalization in ('sym', 'gcn'):
        return gcn_norm(
            edge_index, edge_weight=edge_weight, num_nodes=num_nodes,
            add_self_loops=add_self_loops, dtype=dtype, **kwargs)

    if normalization in ('in-degree', 'out-degree', 'rw'):
        return deg_norm(
            edge_index, edge_weight=edge_weight, num_nodes=num_nodes,
            add_self_loops=add_self_loops, dtype=dtype)

    if normalization in ('in-degree-sym', 'sym-var'):
        return inv_norm(
            edge_index, edge_weight=edge_weight, num_nodes=num_nodes,
            add_self_loops=add_self_loops, dtype=dtype)

    raise AssertionError


def deg_norm(edge_index: Adj, edge_weight: Optional[Tensor] = None, num_nodes: Optional[int] = None,
             add_self_loops: bool = True, dtype: Optional[Any] = None,
             normalization: str = 'in-degree') -> Adj:
    """degree normalization

    Args:
        edge_index (Adj): representation of edges in graph
        edge_weight (Optional[Tensor], optional): optional tensor of edge weights. Defaults to None.
        num_nodes (Optional[int], optional): number of nodes. Defaults to None.
        add_self_loops (bool, optional): flag to add self-loops to edges. Defaults to True.
        dtype (Optional[Any], optional): dtype . Defaults to None.
        normalization (str, optional): ['in-degree', 'out-degree', 'rw']. Defaults to 'sym'.

    Raises:
        AssertionError: raised if unsupported normalization is passed to the function

    Returns:
        Adj: normalized adjacency matrix
    """

    fill_value = 1.0

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)

        if normalization == 'in-degree':

            in_deg = sum(adj_t, dim=0)
            in_deg_inv_sqrt = in_deg.pow_(-0.5)
            in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0.)
            # A = D_in^-1 * A
            adj_t = mul(adj_t, in_deg_inv_sqrt.view(1, -1))

        elif normalization in ('out-degree', 'rw'):
            out_deg = sum(adj_t, dim=1)
            out_deg_inv_sqrt = out_deg.pow_(-0.5)
            out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0.)
            # A = A * D_out^-1
            adj_t = mul(adj_t, out_deg_inv_sqrt.view(-1, 1))

        else:
            raise AssertionError

        return adj_t

    num_nodes = tu.num_nodes.maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1), ),
            dtype=dtype,
            device=edge_index.device)

    if add_self_loops:
        edge_index, edge_weight = tu.add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index[0], edge_index[1]

    if normalization == 'in-degree':
        in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        in_deg_inv = 1.0 / in_deg
        in_deg_inv.masked_fill_(in_deg_inv == float('inf'), 0)
        edge_weight = in_deg_inv[col] * edge_weight

    elif normalization in ('out-degree', 'rw'):
        out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        out_deg_inv = 1.0 / out_deg
        out_deg_inv.masked_fill_(out_deg_inv == float('inf'), 0)
        edge_weight = out_deg_inv[row] * edge_weight

    else:
        raise AssertionError

    return edge_index, edge_weight


def gcn_norm(edge_index: Adj, edge_weight: Optional[Tensor] = None, num_nodes: Optional[int] = None,
             improved: bool = False, add_self_loops: bool = True, dtype: Optional[Any] = None) -> Adj:
    """gcn-like normalization of adjacency matrix

    Args:
        edge_index (Adj): representation of edges in graph
        edge_weight (Optional[Tensor], optional): optional tensor of edge weights. Defaults to None.
        num_nodes (Optional[int], optional): number of nodes. Defaults to None.
        improved (bool, optional): whether or whether not to use improved normalization (weighting self-loops twice). Defaults to False.
        add_self_loops (bool, optional): flag to add self-loops to edges. Defaults to True.
        dtype (Optional[Any], optional): dtype . Defaults to None.

    Returns:
        Adj: normalized adjacency matrix
    """


    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)

        in_deg = sum(adj_t, dim=1)
        #out_deg = sum(adj_t, dim=0)

        in_deg_inv_sqrt = in_deg.pow_(-0.5)
        # out_deg_inv_sqrt = out_deg.pow_(-0.5)

        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0.)
        # out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0.)
        # A = D_in^-0.5 * A * D_out^-0.5
        adj_t = mul(adj_t, in_deg_inv_sqrt.view(1, -1))
        adj_t = mul(adj_t, in_deg_inv_sqrt.view(-1, 1))

        return adj_t

    num_nodes = tu.num_nodes.maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = tu.add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]

    out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    # out_deg_inv_sqrt = out_deg.pow_(-0.5)

    # out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0)
    # A = D_in^-0.5 * A * D_out^-0.5
    edge_weight = in_deg_inv_sqrt[col] * edge_weight * in_deg_inv_sqrt[row]

    return edge_index, edge_weight


def inv_norm(edge_index: Adj, edge_weight: Optional[Tensor] = None, num_nodes: Optional[int] = None,
             add_self_loops: bool = True, dtype: Optional[Any] = None) -> Adj:
    """normalization layer with symmetric inverse-degree normalization

    Args:
        edge_index (Adj): representation of edges in graph
        edge_weight (Optional[Tensor], optional): optional tensor of edge weights. Defaults to None.
        num_nodes (Optional[int], optional): number of nodes. Defaults to None.
        add_self_loops (bool, optional): flag to add self-loops to edges. Defaults to True.
        dtype (Optional[Any], optional): dtype . Defaults to None.

    Returns:
        Adj: normalized adjacency matrix
    """

    fill_value = 1.0

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)

        in_deg = sum(adj_t, dim=1)
        in_deg_inv = in_deg.pow_(-1.0)
        in_deg_inv.masked_fill_(in_deg_inv == float('inf'), 0.)
        adj_t = mul(adj_t, in_deg_inv.view(1, -1))
        adj_t = mul(adj_t, in_deg_inv.view(-1, 1))

        return adj_t

    num_nodes = tu.num_nodes.maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = tu.add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]

    out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    in_deg_inv = in_deg.pow_(-1.0)
    in_deg_inv.masked_fill_(in_deg_inv == float('inf'), 0)
    # A = D_in^-1 * A * D_out^-1
    edge_weight = in_deg_inv[col] * edge_weight * in_deg_inv[row]

    return edge_index, edge_weight


class PropagationChain(nn.Module):
    """convenience layer which allows creation of a list chain of propagations (similar to torch.nn.Sequential)"""

    def __init__(self, propagations: List[callable], activations: Optional[List[callable]] = None):
        super().__init__()
        self.propagations = propagations
        self.activations = activations

    def forward(self, x, edge_index, **kwargs):
        h = x
        for i, p in enumerate(self.propagations):
            h = p(h, edge_index=edge_index, **kwargs)
            if self.activations is not None:
                act = self.activations[i]
                h = act(h)

        return h


class GraphIdentity(nn.Module):
    """simple no-op layer compatible with API of typical graph-convolutional layers"""
    def __init__(self, *_, **__):
        super().__init__()

    def forward(self, x: Tensor, *_, **__) -> Tensor:
        return x


class ConnectedComponents(MessagePassing):
    """layer finding connected components of a graph"""
    def __init__(self):
        super().__init__(aggr="max")

    def forward(self, data):
        x = torch.arange(data.num_nodes).view(-1, 1)
        last_x = torch.zeros_like(x)

        while not x.equal(last_x):
            last_x = x.clone()
            x = self.propagate(data.edge_index, x=x)
            x = torch.max(x, last_x)

        unique, perm = torch.unique(x, return_inverse=True)
        perm = perm.view(-1)

        if "batch" not in data:
            return unique.size(0), perm

        cc_batch = unique.scatter(dim=-1, index=perm, src=data.batch)
        return cc_batch.bincount(minlength=data.num_graphs), perm

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class APPNPPropagation(MessagePassing):
    """APPNP-like propagation (approximate personalized page-rank)
    code taken from the torch_geometric repository on GitHub (https://github.com/rusty1s/pytorch_geometric)
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 normalization: str = 'sym', **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        assert normalization in ('sym', 'rw', 'in-degree', 'out-degree', None)
        self.normalization = normalization

        self._cached_edge_index = None
        self._cached_adj_t = None

    def clear_and_disable_cache(self):
        """ Clears and disables the cache. """
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.cached = False
        logging.info(f'GPN-APPNP disabled cache.')

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalization is not None:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = mat_norm(
                        edge_index, edge_weight, x.size(self.node_dim), improved=False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype,
                        normalization=self.normalization
                    )

                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)

                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:

                    edge_index = mat_norm(
                        edge_index, edge_weight, x.size(self.node_dim), improved=False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype,
                        normalization=self.normalization
                    )

                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x

        for _ in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)

                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)

            x = x * (1 - self.alpha)
            x += self.alpha * h

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)

import numpy as np
import torch
from data.util import vertex_intersection

@torch.no_grad()
def inductive_feature_shift(model, data_before, data_after):
    """ Measures the shift of vertex features when adding new vertices to the graph. 
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to evaluate the shift of.
    data_before : torch_geometric.data.Data
        Graph before edges and vertices are added.
    data_after : torch_geometric.data.Data
        Graph after edges and verties are added. Should be a supergraph of the original graph.
    
    Returns:
    --------
    shift : torch.Tensor, shape [N_intersection]
        Shifts for vertices that are in the graph before and after.
    idx_intersection_before : torch.Tensor, shape [N_intersection]
        Idxs of these vertices in `data_before`.
    idx_intersection_after : torch.Tensor, shape [N_intersection]
        Idxs of these vertices in `data_after`.
    """
    assert len(set(data_before.vertex_to_idx.keys()) - set(data_after.vertex_to_idx.keys())) == 0, f'Data graph after is not a superset of data graph before!'

    model.eval()
    h_before = model(data_before)[-1].cpu()
    h_after = model(data_after)[-1].cpu()

    idx_intersection_before, idx_intersection_after = vertex_intersection(data_before.cpu(), data_after.cpu())
    shift = (h_before[idx_intersection_before] - h_after[idx_intersection_after]).norm(dim=1)
    return shift, idx_intersection_before, idx_intersection_after
    # return shift[data_before.mask[idx_intersection_before]].numpy(), shift[~data_before.mask[idx_intersection_before]].numpy()
import numpy as np
import torch
import os.path as osp
from collections import defaultdict
from warnings import warn
from torch_geometric.data import Dataset, Data
import gust.datasets

class GustDataset(Dataset):
    """ Dataset derived from the gust library. """

    def __init__(self, name, make_undirected=True, make_unweighted=True, select_lcc=True, remove_self_loops=True):
        """ Initializes the gust-based dataset. 
        
        Parameters:
        -----------
        name : str
            The key in the gust database.
        make_undirected : bool
            If the graph is made undirected.
        make_unweighted : bool
            If the graph is made unweighted.
        select_lcc : bool
            If the lcc of the graph is selected.
        remove_self_loops : bool
            If True, self-loops are removed.
        """
        self.name = name
        self.make_undirected = make_undirected
        self.make_unweighted = make_unweighted
        self.select_lcc = select_lcc
        self.remove_self_loops = remove_self_loops
        sparse_graph = gust.datasets.load_dataset(name).standardize(
            make_unweighted = self.make_unweighted, make_undirected = self.make_undirected,
            no_self_loops = self.remove_self_loops, select_lcc = self.select_lcc
        )
        edge_index = np.array(sparse_graph.adj_matrix.nonzero())
        x = sparse_graph.attr_matrix.todense()
        y = sparse_graph.labels
        if sparse_graph.node_names is not None:
            vertex_to_idx = {vertex_id : idx for idx, vertex_id in enumerate(sparse_graph.node_names)}
        else:
            vertex_to_idx = {f'vertex_{idx}' : idx for idx in range(x.shape[0])}
        if sparse_graph.class_names is not None:
            label_to_idx = {label : idx for idx, label in enumerate(sparse_graph.class_names)}
        else:
            label_to_idx = {f'class_{idx}' : idx for idx in set(y)}
        data = Data(x=torch.tensor(x), y=torch.tensor(y), edge_index=torch.tensor(edge_index))
        data.vertex_to_idx = vertex_to_idx
        data.label_to_idx = label_to_idx
        self.data = data

    
    def __getitem__(self, idx):
        assert idx == 0, f'Gust {self.name} dataset has only one graph'
        return self.data

    def __len__(self, idx):
        return 1

if __name__ == '__main__':
    cora = GustDataset('cora_ml')[0]
    print(cora.x.size(), cora.edge_index.size(), cora.y.size())

    citeseer = GustDataset('citeseer')[0]
    print(citeseer.x.size(), citeseer.edge_index.size(), citeseer.y.size())

    pubmed = GustDataset('pubmed')[0]
    print(pubmed.x.size(), pubmed.edge_index.size(), pubmed.y.size())
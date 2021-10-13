import numpy as np
import torch
import os.path as osp
from collections import defaultdict
from warnings import warn
from torch_geometric.data import Dataset, download_url, Data
import torch_geometric.transforms as T
import tarfile
from transform import NormalizeGraph

dataset_dir = '.datasets'

def parse_linqs_tab_dataset(content_file, cites_file, print_warnings=True):
    """ Parses a LINQs citation dataset database in which content is separated by tabs.
    
    Parameters:
    -----------
    content_file : str or path-like
        A file in which each line describes node features in the format <vertex_id> \t {<attribute> \t} + <class_label>
    cites_file : str or path-like
        A file in which each line describes a link in the format <target_vertex_id> \t <source_vertex_id>
    print_warnings : bool
        If True, warnings are outputted. Default: True

    Returns:
    --------
    attributes : ndarray, shape [N, D]
        Attribute matrix of the graph.
    labels : ndarray, shape [N]
        Labels of the vertices.
    edge_idxs : ndarray, shape [2, E]
        Endpoints of each edge.
    vertex_to_idx : dict
        Mapping from a unique vertex identifier to its index in the numerical ndarrays.
    label_to_idx : dict
        Mapping from a class label to its numerical value in the 'labels' ndarray.
    """
    attributes, classes = dict(), dict()
    with open(content_file) as f:
        for line in f.read().splitlines():
            tokens = line.split('\t')
            attributes[tokens[0]] = list(map(int, tokens[1:-1]))
            classes[tokens[0]] = tokens[-1]
    vertex_to_idx = {v : i for i, v in enumerate(set(attributes.keys()))}
    label_to_idx = {l : i for i, l in enumerate(set(classes.values()))}
    x = [None for _ in range(len(vertex_to_idx))]
    labels = [None for _ in range(len(vertex_to_idx))]
    for v, idx in vertex_to_idx.items():
        x[idx] = attributes[v]
        labels[idx] = label_to_idx[classes[v]]
    edge_idxs = []
    unresolved_links = 0
    with open(cites_file) as f:
        for line in f.read().splitlines():
            u, v = line.split('\t')
            if u in vertex_to_idx and v in vertex_to_idx:
                edge_idxs.append([vertex_to_idx[v], vertex_to_idx[u]])
            else:
                unresolved_links += 1
    if print_warnings and unresolved_links > 0:
        warn(f'{unresolved_links} links could not be resolved because either target or source are not in the content.')
    
    x = np.array(x, dtype=int)
    labels = np.array(labels, dtype=int)
    edge_idxs = np.array(edge_idxs, dtype=int).T
    return x, labels, edge_idxs, vertex_to_idx, label_to_idx

def parse_pubmed(node_file, cites_file):
    """ Parses the Pubmed dataset in the LINQs citation dataset database.
    
    Parameters:
    -----------
    node_file : str or path-like
        A file in which lines 2, 3, ... describe node features in the format <vertex_idx> \t {<attr_name>=<attr_value> \t}+ <summary>
    cites_file : str or path-like
        A file in which each line describes a link in the format <link_idx> \t <source_vertex_idx> \t | \t <target_vertex_idx>

    Returns:
    --------
    attributes : ndarray, shape [N, D]
        Attribute matrix of the graph.
    labels : ndarray, shape [N]
        Labels of the vertices.
    edge_idxs : ndarray, shape [2, E]
        Endpoints of each edge.
    vertex_to_idx : dict
        Mapping from a unique vertex identifier to its index in the numerical ndarrays.
    label_to_idx : dict
        Mapping from a class label to its numerical value in the 'labels' ndarray.
    """
    attributes = defaultdict(dict)
    labels = dict()
    with open(node_file) as f:
        for num, line in enumerate(f.read().splitlines()):
            tokens = line.split('\t')
            if num == 1: # Parse attribute header
                tokens = [token.split(':') for token in tokens]
                assert tokens[0][1] == 'label', f'Expected label field first in PubMed attributes but have {tokens[0][1]}'
                attribute_to_idx = {attr : idx for idx, attr in enumerate(token[1] for token in tokens[1:-1])}
                assert len(attribute_to_idx) == 500, f'Expected 500 attributes in PubMed attributes, but have {len(attribute_to_idx)}'
            elif num > 1:
                for text in tokens[1:]:
                    attr, value = text.split('=')
                    if attr in attribute_to_idx: 
                        attributes[tokens[0]][attr] = float(value)
                    elif attr == 'label':
                        labels[tokens[0]] = value
                    elif attr == 'summary':
                        continue
                    else:
                        raise RuntimeError(f'Unexpected node attribute {attr}')
    x = np.zeros((len(attributes), len(attribute_to_idx)), dtype=float)
    y = np.zeros((len(attributes)), dtype=int)
    vertex_to_idx = {vertex : idx for idx, vertex in enumerate(set(attributes.keys()))}
    label_to_idx = {label : idx for idx, label in enumerate(set(labels.values()))}
    for vertex, attrs in attributes.items():
        for attr, value in attrs.items():
            x[vertex_to_idx[vertex], attribute_to_idx[attr]] = value
            y[vertex_to_idx[vertex]] = label_to_idx[labels[vertex]]

    edge_idxs = []
    unknown = [] # Unique identifiers, that are however not consecutive...
    with open(cites_file) as f:
        for num, line in enumerate(f.read().splitlines()):
            tokens = line.split('\t')
            if num > 1:
                assert tokens[2] == '|', f"Expected '|' character in line {num}, but got '{tokens[2]}'"
                unknown.append(int(tokens[0]))
                u, v = tokens[1], tokens[3]
                assert u.startswith('paper:') and v.startswith('paper:')
                u, v = u.replace('paper:', ''), v.replace('paper:', '')
                assert u in vertex_to_idx, f"Unknown node id {u} in line {num}"
                assert v in vertex_to_idx, f"Unknown node id {v} in line {num}"
                edge_idxs.append([vertex_to_idx[u], vertex_to_idx[v]])
    edge_idxs = np.array(edge_idxs, dtype=int).T
    return x, y, edge_idxs, vertex_to_idx, label_to_idx

class Cora(Dataset):
    def __init__(self, root=osp.join(dataset_dir, 'cora'), transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [osp.join('cora','cora.content'), osp.join('cora','cora.cites'), ]

    @property
    def processed_file_names(self):
        return ['cora.pt']

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url('https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz', self.raw_dir)
        print('Extracting...')
        with tarfile.open(osp.join(self.raw_dir, 'cora.tgz'), "r:gz") as tar:
            tar.extractall(path=self.raw_dir)
            tar.close()

    def process(self):
        x, y, edge_idxs, vertex_to_idx, label_to_idx = parse_linqs_tab_dataset(osp.join(self.raw_dir, self.raw_file_names[0]), osp.join(self.raw_dir, self.raw_file_names[1]))
        data = Data(x=torch.tensor(x), y=torch.tensor(y), edge_index=torch.tensor(edge_idxs))
        data.vertex_to_idx = vertex_to_idx
        data.label_to_idx = label_to_idx
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names[0]))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        assert idx == 0
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[0]))
        return data

class Citeseer(Dataset):
    def __init__(self, root=osp.join(dataset_dir, 'citeseer'), transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [osp.join('citeseer','citeseer.content'), osp.join('citeseer','citeseer.cites'), ]

    @property
    def processed_file_names(self):
        return ['citeseer.pt']

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url('https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz', self.raw_dir)
        print('Extracting...')
        with tarfile.open(osp.join(self.raw_dir, 'citeseer.tgz'), "r:gz") as tar:
            tar.extractall(path=self.raw_dir)
            tar.close()

    def process(self):
        x, y, edge_idxs, vertex_to_idx, label_to_idx = parse_linqs_tab_dataset(osp.join(self.raw_dir, self.raw_file_names[0]), osp.join(self.raw_dir, self.raw_file_names[1]))
        data = Data(x=torch.tensor(x), y=torch.tensor(y), edge_index=torch.tensor(edge_idxs))
        data.vertex_to_idx = vertex_to_idx
        data.label_to_idx = label_to_idx
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names[0]))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        assert idx == 0
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[0]))
        return data

class Pubmed(Dataset):
    def __init__(self, root=osp.join(dataset_dir, 'pubmed'), transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [osp.join('Pubmed-Diabetes', 'data', 'Pubmed-Diabetes.NODE.paper.tab'), osp.join('Pubmed-Diabetes', 'data', 'Pubmed-Diabetes.DIRECTED.cites.tab'), ]

    @property
    def processed_file_names(self):
        return ['pubmed.pt']

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url('https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz', self.raw_dir)
        print('Extracting...')
        with tarfile.open(osp.join(self.raw_dir, 'Pubmed-Diabetes.tgz'), "r:gz") as tar:
            tar.extractall(path=self.raw_dir)
            tar.close()

    def process(self):
        x, y, edge_idxs, vertex_to_idx, label_to_idx = parse_pubmed(osp.join(self.raw_dir, self.raw_file_names[0]), osp.join(self.raw_dir, self.raw_file_names[1]))
        data = Data(x=torch.tensor(x), y=torch.tensor(y), edge_index=torch.tensor(edge_idxs))
        data.vertex_to_idx = vertex_to_idx
        data.label_to_idx = label_to_idx
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names[0]))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        assert idx == 0
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[0]))
        return data

if __name__ == '__main__':
    transforms = T.Compose([T.ToUndirected(), NormalizeGraph(min_class_count = 20 / 0.05, verbose = True)])

    d = Cora(transform=transforms)[0]
    print(d.x, d.y, d.edge_index)

    d = Citeseer(transform=transforms)[0]
    print(d.x, d.y, d.edge_index)

    d = Pubmed(transform=transforms)[0]
    print(d.x, d.y, d.edge_index)
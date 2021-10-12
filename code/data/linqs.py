import numpy as np
import torch
import os
from collections import defaultdict
from warnings import warn

class TabDataset:
    """ Dataset from LINQs citation dataset database in which content is separated by tabs.

    Each dataset has the following files:
    - content_file : Each line describes nodes features in the following format <vertex_id> \t [<attribute> \t] + <class_label>
    - cites_file : Each line describes a link in the following format <target_vertex_id> \t <source_vertex_id>
    
    Downloaded from: https://linqs.soe.ucsc.edu/data
    """

    def __init__(self, content_file, cites_file, print_warnings=True):

        attributes, classes = dict(), dict()
        with open(content_file) as f:
            for line in f.read().splitlines():
                tokens = line.split('\t')
                attributes[tokens[0]] = list(map(int, tokens[1:-1]))
                classes[tokens[0]] = tokens[-1]
        self.vertex_to_idx = {v : i for i, v in enumerate(attributes.keys())}
        self.label_to_idx = {l : i for i, l in enumerate(classes.values())}
        self.attributes = [None for _ in range(len(self.vertex_to_idx))]
        self.labels = [None for _ in range(len(self.vertex_to_idx))]
        for v, idx in self.vertex_to_idx.items():
            self.attributes[idx] = attributes[v]
            self.labels[idx] = self.label_to_idx[classes[v]]
        edge_idxs = []
        unresolved_links = 0
        with open(cites_file) as f:
            for line in f.read().splitlines():
                u, v = line.split('\t')
                if u in self.vertex_to_idx and v in self.vertex_to_idx:
                    edge_idxs.append([self.vertex_to_idx[v], self.vertex_to_idx[u]])
                else:
                    unresolved_links += 1
        if print_warnings and unresolved_links > 0:
            warn(f'{unresolved_links} links could not be resolved because either target or source are not in the content.')
        
        self.attributes = np.array(self.attributes, dtype=int)
        self.labels = np.array(self.labels, dtype=int)
        self.edge_idxs = np.array(edge_idxs, dtype=int).T

class PubmedDataset:
    """ Pubmed from LINQs citation dataset database. 

    Downloaded from: https://linqs.soe.ucsc.edu/data"""

    def __init__(self, node_file, cites_file):
        attributes = defaultdict(dict)
        labels = dict()
        with open(node_file) as f:
            for num, line in enumerate(f.read().splitlines()):
                tokens = line.split('\t')
                if num == 1: # Parse attribute header
                    tokens = [token.split(':') for token in tokens]
                    assert tokens[0][1] == 'label', f'Expected label field first in PubMed attributes but have {tokens[0][1]}'
                    self.attribute_to_idx = {attr : idx for idx, attr in enumerate(token[1] for token in tokens[1:-1])}
                    assert len(self.attribute_to_idx) == 500, f'Expected 500 attributes in PubMed attributes, but have {len(self.attribute_to_idx)}'
                elif num > 1:
                    for text in tokens[1:]:
                        attr, value = text.split('=')
                        if attr in self.attribute_to_idx: 
                            attributes[tokens[0]][attr] = float(value)
                        elif attr == 'label':
                            labels[tokens[0]] = value
                        elif attr == 'summary':
                            continue
                        else:
                            raise RuntimeError(f'Unexpected node attribute {attr}')
        self.label_to_idx = {label : idx for idx, label in enumerate(labels.values())}
        self.attributes = np.zeros((len(attributes), len(self.attribute_to_idx)), dtype=float)
        self.labels = np.zeros((len(attributes)), dtype=int)
        self.vertex_to_idx = {vertex : idx for idx, vertex in enumerate(attributes.keys())}
        for vertex, attrs in attributes.items():
            for attr, value in attrs.items():
                self.attribute_to_idx[self.vertex_to_idx[vertex], self.attribute_to_idx[attr]] = value
                self.labels[self.vertex_to_idx[vertex]] = self.label_to_idx[labels[vertex]]

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
                    assert u in self.vertex_to_idx, f"Unknown node id {u} in line {num}"
                    assert v in self.vertex_to_idx, f"Unknown node id {v} in line {num}"
                    edge_idxs.append([self.vertex_to_idx[u], self.vertex_to_idx[v]])
        self.edge_idxs = np.array(edge_idxs, dtype=int).T

if __name__ == '__main__':
    d = PubmedDataset(os.path.join('data','Pubmed-Diabetes','data', 'Pubmed-Diabetes.NODE.paper.tab'), os.path.join('data', 'Pubmed-Diabetes','data','Pubmed-Diabetes.DIRECTED.cites.tab'))
    print(d.attributes.shape, d.labels.shape, d.edge_idxs.shape)

    d = TabDataset(os.path.join('data/citeseer/citeseer.content'), os.path.join('data/citeseer/citeseer.cites'))
    print(d.attributes.shape, d.labels.shape, d.edge_idxs.shape)

    d = TabDataset(os.path.join('data/cora/cora.content'), os.path.join('data/cora/cora.cites'))
    print(d.attributes.shape, d.labels.shape, d.edge_idxs.shape)

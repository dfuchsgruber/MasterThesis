from typing import Any, Mapping
import configuration
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from data.base import SingleGraphDataset
import data.util
import data.constants as dconst
import os.path as osp
import os
from util import sparse_max, get_cache_path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import logging
import util

from transformers import AutoModel, AutoTokenizer

def make_mapping_collatable(d: Mapping[str, Any]) -> Mapping[str, Any]:
    """ Fixes the keys in a mapping so that they do not contain keys the tg collator is senstitive to.
    That includes the substrings 'batch', 'index', 'face' (yes, very stupid...)
    """
    result = {}
    for k, v in d.items():
        old_k, k = k, util.make_key_collatable(k)
        if k in result:
            raise RuntimeError(f'Duplicate keys {k} by sanitizing.')
        if k != old_k:
            logging.info(f'Making mapping collatable: Sanitized "{old_k}" to "{k}"')
        result[k] = v
    return result

def load_embedded_word_features(dataset_name, language_model, cache_dir=osp.join(get_cache_path(), 'npz-word-features'), max_length=512):
    """
    Loads embedded word features for a given Npz dataset and language model.

    Parameters:
    -----------
    dataset_name : str
        The dataset name. Must be key in `NpzDataset.raw_files`
    language_model : str
        The language model to use.
    cache_dir : str
        The cache directory.
    max_length : int
        The maximum sequence length for the model to handle. Sequences are truncated if longer.
    
    Returns:
    --------
    X : ndarray, shape [N_raw, D]
        The embedded texts.
    feature_to_idx : dict
        A mapping that names every column in [0, ... D-1].
    """ 
    cache_file = osp.join(cache_dir, f'{dataset_name}:{language_model}:word-embeddings.npy')
    os.makedirs(osp.dirname(cache_file), exist_ok=True)
    if osp.exists(cache_file):
        X =  np.load(cache_file)
    else:
        loader = np.load(NpzDataset.raw_files[dataset_name], allow_pickle=True)
        if 'attr_text' in loader:
            corpus = loader['attr_text']
        else:
            raise RuntimeError(f'Npz Dataset {dataset_name} has no text attribute "attr_text" to embed.')
        
        if language_model in ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', ):
            sentences = corpus.tolist()
            model = SentenceTransformer(language_model)
            if torch.cuda.is_available():
                model = model.cuda()
            X = model.encode(sentences)
        else:
            # Create embeddings using any `transformer` pre-trained model
            tokenizer = AutoTokenizer.from_pretrained(language_model, max_length=max_length, padding=True)
            model = AutoModel.from_pretrained(language_model).cuda()
            if torch.cuda.is_available():
                model = model.cuda()

            embedded = []
            with torch.no_grad():
                for idx in tqdm(range(corpus.shape[0]), desc=f'Embedding using {language_model}'):
                    inputs = {k : v[..., :max_length] for k, v in tokenizer(corpus[idx], return_tensors='pt').items()}
                    if torch.cuda.is_available():
                        inputs = {k : v.cuda() for k, v in inputs.items()}
                    output = model(**inputs)
                    embedded.append(output.pooler_output.cpu().clone())
                    del output

            X = torch.cat(embedded, dim=0).numpy() # shape (N, embedding_size)

        np.save(cache_file, X)
    
    feature_to_idx = {f'feature_{i}' : i for i in range(X.shape[1])}
    return X, feature_to_idx

class NpzDataset(SingleGraphDataset):
    
    raw_files = {
        dconst.CORA_FULL : osp.join('data', 'raw', 'cora_full.npz'),
        dconst.CORA_ML : osp.join('data', 'raw', 'cora_ml.npz'),
        dconst.CITESEER : osp.join('data', 'raw', 'citeseer.npz'),
        dconst.PUBMED : osp.join('data', 'raw', 'pubmed.npz'),
        dconst.DBLP : osp.join('data', 'raw', 'dblp.npz'),
        dconst.OGBN_ARXIV : osp.join('data', 'raw', 'ogbn-arxiv.npz'),
        dconst.COAUTHOR_CS : osp.join('data', 'raw', 'ms_academic_cs_with_text.npz'),
        dconst.COAUTHOR_PHYSICS :  osp.join('data', 'raw', 'ms_academic_phy_with_text.npz'),
        dconst.AMAZON_COMPUTERS : osp.join('data', 'raw', 'amazon_electronics_computers.npz'),
        dconst.AMAZON_PHOTO : osp.join('data', 'raw', 'amazon_electronics_photo.npz'),
    }

    @classmethod
    def list_datasets():
        return list(NpzDataset.raw_files.keys())
    
    @staticmethod
    def _load_graph(loader, make_undirected=True, make_unweighted=True, select_lcc=True, remove_self_loops=True):
        """ Loads and preprocessed the graph adjacency. 
        
        Parameters:
        -----------
        loader : npz.array
            Npz array with all the information.
        make_undirected : bool
            If the graph should be symmetric. Is done by taking the max of A and A.T
        make_unweighted : bool
            Replaces the edge weights with 1.0
        select_lcc : bool
            Selects only the largest selected component.
        remove_self_loops : bool
            Removes self-loops from the graph.

        Returns:
        --------
        A : sp.csr_matrix
            The adjacency matrix
        vertices_to_keep : ndarray, shape [N_raw]
            A mask that tells which vertices of the original raw data to keep for the data.
        """
        
        adj_data = loader['adj_data']
        
        # Make unweighted : Replace data with ones
        if make_unweighted:
            adj_data = np.ones_like(adj_data)
        A = sp.csr_matrix((adj_data, loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])
        
        # Make undirected : Maximum of A and its transpose
        if make_undirected:
            A = sparse_max(A, A.T)
        edge_index = np.array(A.nonzero())
        
        # Remove self-loops
        if remove_self_loops:
            is_self_loop = edge_index[0, :] == edge_index[1, :]
            edge_index = edge_index[:, ~is_self_loop]
            
        # Select only the lcc from A
        if select_lcc:
            n_components, labels = sp.csgraph.connected_components(A)
            label_names, label_counts = np.unique(labels, return_counts=True)
            label_lcc = label_names[label_counts.argmax()]
            
            # Only keep vertices with labels == label_lcc
            vertices_to_keep = labels == label_lcc
            A = A.tocsr()[vertices_to_keep].tocsc()[:, vertices_to_keep].tocsr()
        else:
            vertices_to_keep = np.ones(A.shape[0], dtype=bool)
            
        return A, vertices_to_keep
    
    @staticmethod
    def _build_vectorizer(corpus, y, idx_to_label, corpus_labels='all', min_token_frequency=10, normalize='l2', vectorizer='tf-idf'):
        """ Builds and fits a vectorizer on all elements 
        
        Parameters:
        -----------
        corpus : ndarray, shape [N], dtype=UXX
            The corpus to fit.
        y : ndarray, shape [N]
            Labels for each vertex.
        idx_to_label : dict
            Mapping from idx to a vertex label
        corpus_labels : 'all' or iterable
            Which class labels to base the vectorizer on. This prevents data leakage.
        min_token_frequency : int
            How often a token must appear to at minimum to be recognized by the vectorizer
        normalize : 'l1', 'l2' or None
            How the bag of words features are normalized.

        Returns:
        --------
        vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The vectorizer for the corpus.
        """
        
        if corpus_labels == 'all':
            corpus_labels = set(idx_to_label.values())
        else:
            corpus_labels = set(corpus_labels)
        has_corpus_label = np.array([idx_to_label[label] in corpus_labels for label in y])
        if vectorizer.lower() in ('tf-idf', 'tfidf'):
            vectorizer = TfidfVectorizer(min_df=min_token_frequency, norm=normalize)
        elif vectorizer.lower() in ('count', ):
            vectorizer = CountVectorizer(min_df=min_token_frequency)
        else:
            raise RuntimeError(f'Unsupported vectorizer type {vectorizer}.')
        vectorizer.fit(corpus[has_corpus_label])
        return vectorizer
    
    @staticmethod
    def _build_vertex_to_idx(idx_to_vertex, vertices_to_keep):
        """ Builds the mapping from vertex -> idx. 
        
        Parameters:
        -----------
        idx_to_vertex : dict
            Mapping from int -> vertex name
        vertices_to_keep : ndarray, shape [N]
            Which vertices in the original data (with N_raw >= N vertices) to keep.

        Returns:
        --------
        vertex_to_idx : dict
            Mapping from vertex name to its index in X
        """
        
        vertex_names = [None for _ in range(max(idx_to_vertex.keys()) + 1)]
        for idx, name in idx_to_vertex.items():
            vertex_names[idx] = name
        assert None not in vertex_names, f'No name specified for vertex at idx {vertex_names.index(None)}'
        vertex_names = np.array(vertex_names)[vertices_to_keep]
        vertex_to_idx = {vertex : idx for idx, vertex in enumerate(vertex_names)}
        return vertex_to_idx

    @staticmethod
    def _vectorize(corpus, vectorizer):
        """ Loads the attribute matrix of the dataset using a vectorizer. 
        
        Parameters:
        -----------
        corpus : ndarray, shape [N]
            The text attributes of each vertex.
        vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The vectorizer.
        
        Returns:
        --------
        X : sp.sparse_matrix, shape [N, vocab_size]
            Attribute matrix
        feature_to_idx : dict
            Mapping from feature name (word in vocabulary) to idx in X.
        """
        X = vectorizer.transform(corpus)
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            feature_names = vectorizer.get_feature_names()
        feature_to_idx = {feat : idx for idx, feat in enumerate(feature_names)}
        return X, feature_to_idx

    @staticmethod
    def build(config: configuration.DataConfiguration, transform=None):
        if config.dataset not in NpzDataset.raw_files:
            raise ValueError(f'Npz dataset {config.dataset} does not exist.')
        loader = np.load(NpzDataset.raw_files[config.dataset], allow_pickle=True)
        A, vertices_to_keep = NpzDataset._load_graph(loader, make_undirected=True, select_lcc=True, remove_self_loops=False)
        logging.info('Data Loading - Loaded adjacency matrix.')
        y = loader['labels'][vertices_to_keep]

        if 'idx_to_class' in loader:
            idx_to_label = loader['idx_to_class'].item()
        elif 'class_names' in loader:
            idx_to_label = {idx : name for idx, name in enumerate(loader['class_names'])}
        else:
            logging.info(f'Data Loading - Did not find class names in {config.dataset}. Generating default names...')
            idx_to_label = {idx : f'class_{idx}' for idx in range(int(max(y)) + 1)}

        if 'idx_to_node' in loader:
            idx_to_node = loader['idx_to_node'].item()
        elif 'node_names' in loader:
            idx_to_node = {idx : name for idx, name in enumerate(loader['node_names'])}
        else:
            logging.info(f'Data Loading - Did not find vertex names in {config.dataset}. Generating default names...')
            idx_to_node = {idx : f'node_{idx}' for idx in range(vertices_to_keep.shape[0])}

        # Build the attribute matrix
        if config.preprocessing == 'bag_of_words':
            corpus = loader['attr_text']
            vectorizer = NpzDataset._build_vectorizer(corpus[vertices_to_keep], y, idx_to_label, corpus_labels=config.corpus_labels, 
                min_token_frequency=config.min_token_frequency, normalize=config.normalize, vectorizer=config.vectorizer)
            X, feature_to_idx = NpzDataset._vectorize(corpus[vertices_to_keep], vectorizer)
            X = X.todense()
            vertex_to_idx = NpzDataset._build_vertex_to_idx(idx_to_node, vertices_to_keep)
        elif config.preprocessing == 'word_embedding':
            corpus = loader['attr_text']
            X, feature_to_idx = load_embedded_word_features(config.dataset, config.language_model)
            X = X[vertices_to_keep]
            vertex_to_idx = NpzDataset._build_vertex_to_idx(idx_to_node, vertices_to_keep)
        elif config.preprocessing is None or config.preprocessing == 'none':
            if 'features' in loader:
                # Dense features
                X = loader['features'][vertices_to_keep]
                vertex_to_idx = NpzDataset._build_vertex_to_idx(idx_to_node, vertices_to_keep)
                feature_to_idx = {f'feature_{i}' : i for i in range(X.shape[1])}
            else:
                # Sparse features
                X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape=loader['attr_shape'])[vertices_to_keep].todense()
                vertex_to_idx = NpzDataset._build_vertex_to_idx(idx_to_node, vertices_to_keep)
            if 'attr_names' in loader and len(loader['attr_names']):
                feature_to_idx = {f'{feature}' : int(idx) for idx, feature in enumerate(loader['attr_names'])}
            elif 'idx_to_attr' in loader:
                feature_to_idx = {f'{attr}' : int(idx) for idx, attr in loader['idx_to_attr'].item().items()}
            else:
                feature_to_idx = {f'feature_{i}' : i for i in range(X.shape[1])}
            assert len(feature_to_idx) == X.shape[1], f'Mismatching sizes for attr_names {len(feature_to_idx)} != {X.shape[1]}'
            assert len(set(feature_to_idx.values())) == X.shape[1]
        else:
            raise ValueError(f'Unknown preprocessing for features of type {config.preprocessing}')
        logging.info('Data Loading - Built attribute matrix.')

        if config.normalize is None or config.normalize == 'none':
            pass
        if config.normalize == 'l1':
            X /= np.linalg.norm(X, axis=1, ord=1, keepdims=True)
        elif config.normalize == 'l2':
            X /= np.linalg.norm(X, axis=1, ord=2, keepdims=True)
        else:
            raise ValueError(f'Unsupported normalization {config.normalize}')

        # Fix a bug with the collator: No 

        X *= config.feature_scale

        label_to_idx = make_mapping_collatable({label : idx for idx, label in idx_to_label.items() if idx in y})
        y, label_to_idx, _ = data.util.compress_labels(y, label_to_idx)
        _data = SingleGraphDataset.build(X, np.array(A.nonzero()), y, 
            make_mapping_collatable(vertex_to_idx), 
            (label_to_idx), np.ones_like(y), transform=transform, 
            feature_to_idx=make_mapping_collatable(feature_to_idx)).data
        
        # Add additional attributes to the data by the dataset
        if config.dataset == dconst.OGBN_ARXIV:
            _data._mask_train = torch.Tensor(loader['mask_train'][vertices_to_keep]).bool()
            _data._mask_val = torch.Tensor(loader['mask_val'][vertices_to_keep]).bool()
            _data._mask_test = torch.Tensor(loader['mask_test'][vertices_to_keep]).bool()
            _data.year = torch.Tensor(loader['year'][vertices_to_keep]).long()

        return NpzDataset(_data, transform=transform)

if __name__ == '__main__':
    pass
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS
from sklearn.manifold import TSNE

class Identity:

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X

class TSNEWrapper:

    def __init__(self, number_components=2, **kwargs):
        self.number_components = number_components
        self.kwargs = kwargs

    def fit(self, X):
        # TSNE is stateless
        pass

    def transform(self, X):
        tsne = TSNE(n_components = self.number_components, **self.kwargs)
        return tsne.fit_transform(X)


class DimensionalityReduction:
    """ Base class for dimensionality reduction. """

    def __init__(self, type='pca', number_components=2, per_class=False, number_neighbours=5, seed=1337, *args, **kwargs):
        self.type = type
        self.number_components = number_components
        self.number_neighbours = number_neighbours
        self.per_class = per_class
        self.seed = seed

    @property
    def compressed_name(self):
        if not self.type or self.type.lower() == 'none' or self.type.lower() == 'identity':
            return 'no'
        else:
            return f'{self.number_components}-{self.type.lower()}'

    def _make_transform(self):
        if not self.type or self.type.lower() == 'none' or self.type.lower() == 'identity':
            return Identity()
        elif self.type.lower() == 'pca':
            return PCA(n_components=self.number_components, random_state=self.seed)
        elif self.type.lower() == 'isomap':
            return Isomap(n_components=self.number_components, n_neighbors=self.number_neighbours)
        elif self.type.lower() == 'tsne':
            return TSNEWrapper(number_components=self.number_components, random_state=self.seed)
        # elif self.type.lower() == 'mds':  # MDS is not a fixed manifold?
        #     return MDS(n_components=self.number_components)
        else:
            raise RuntimeError(f'Unsupported dimensionality reduction {self.type}')

    @torch.no_grad()
    def fit(self, features):
        """ Fits the dimensionality reduction. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        """
        if self.per_class:
            raise NotImplementedError
        else:
            self._transform = self._make_transform()
            self._transform.fit(features.cpu().numpy())

    def transform(self, features):
        """ Applies the dimensionality reduction(s).
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        
        Returns:
        --------
        transformed : dict
            A dict mapping from class-label (int) to the transformed features.
        """
        if self.per_class:
            raise NotImplemented
        else:
            return self._transform.transform(features.cpu().numpy())
        
    def statistics(self):
        """ Returns statistics about the fit. 
        
        Returns:
        --------
        statistics : dict
            A dict with all statistics.
        """
        if self.per_class:
            raise NotImplemented
        else:
            if self.type.lower() == 'pca':
                return {
                    'explained_variance_ratio' : float(self._transform.explained_variance_ratio_.sum()),
                }
            else:
                return {}
        
    def __str__(self):
        return '\n'.join([
            f'\t Type : {self.type}',
            f'\t Number of Components : {self.number_components}', 
        ])
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS


class Identity:

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X

class DimensionalityReduction:
    """ Base class for dimensionality reduction. """

    def __init__(self, type='pca', number_components=2, per_class=False, number_neighbours=5, *args, **kwargs):
        self.type = type
        self.number_components = number_components
        self.number_neighbours = number_neighbours
        self.per_class = per_class

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
            return PCA(n_components=self.number_components)
        elif self.type.lower() == 'isomap':
            return Isomap(n_components=self.number_components, n_neighbors=self.number_neighbours)
        # elif self.type.lower() == 'mds':  # MDS is not a fixed manifold?
        #     return MDS(n_components=self.number_components)
        else:
            raise RuntimeError(f'Unsupported dimensionality reduction {self.type}')

    @torch.no_grad()
    def fit(self, features, labels):
        """ Fits the dimensionality reduction. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        labels : torch.Tensor, shape [N, num_classes]
            Soft class labels. Unused for this approach.
        """
        if self.per_class:
            raise NotImplementedError # For now, if one impelements it should also be in the `compressed_name`
            # TODO: soft labels?
            self.transforms = {}
            labels_hard = labels.argmax(1)
            for label in torch.unique(labels_hard):
                label = label.item()
                transform = self._make_transform()
                transform.fit(features[labels_hard == label].cpu().numpy())
                self.transforms[label] = transform
        else:
            transform = self._make_transform()
            transform.fit(features.cpu().numpy())
            self.transforms = {label : transform for label in range(labels.size(1))}

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
        return {label : transform.transform(features.cpu().numpy()) for label, transform in self.transforms.items()}
        
    def __str__(self):
        return '\n'.join([
            f'\t Type : {self.type}',
            f'\t Number of Components : {self.number_components}', 
        ])
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy.stats
from warnings import warn
from sklearn.decomposition import PCA

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov
    
    From: https://github.com/pytorch/pytorch/issues/19037#issue-430654869
    """
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

class FeatureSpaceDensityGaussianPerClass(torch.nn.Module):
    """ Model that estimates features space density fitting a Gaussian per class. """

    name = 'GaussianPerClass'

    def __init__(self, pca=False, pca_number_components=-1, pca_per_class=False, diagonal_covariance=False, **kwargs):
        super().__init__()
        self._fitted = False
        self.pca = pca
        self.pca_number_components = pca_number_components
        self.pca_per_class = pca_per_class
        self.diagonal_covariance = diagonal_covariance

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t PCA : {self.pca}',
            f'\t PCA number components : {self.pca_number_components}',
            f'\t PCA per class : {self.pca_per_class}',
            f'\t Diagonal covariance : {self.diagonal_covariance}'
        ])

    def _fit_pca(self, features, labels):
        """ Fits the pca to the feature space. If pca isnt used, identity projection is applied.

        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features to transform.
        labels : torch.Tensor, shape [N]
            Labels.
        """
        self.pca_projections = nn.ParameterDict()
        self.pca_means = nn.ParameterDict()

        if self.pca and not self.pca_per_class:
            pca_all = PCA(n_components=self.pca_number_components)
            pca_all.fit(features.cpu().numpy())

        for label in torch.unique(labels):
            name = f'class_{label.item()}'
            # Calculate the pca for this class
            if self.pca:
                if self.pca_per_class:
                    pca = PCA(n_components=self.pca_number_components)
                    pca.fit(features[labels == label].cpu().numpy())
                    self.pca_projections[name] = nn.parameter.Parameter(torch.tensor(pca.components_.T))
                    self.pca_means[name] = nn.parameter.Parameter(torch.tensor(pca.mean_))
                else:
                    self.pca_projections[name] = nn.parameter.Parameter(torch.tensor(pca_all.components_.T))
                    self.pca_means[name] = nn.parameter.Parameter(torch.tensor(pca_all.mean_))
            else:
                self.pca_projections[name] = nn.parameter.Parameter(torch.eye(features.size()[1]))
                self.pca_means[name] = nn.parameter.Parameter(torch.zeros(features.size()[1]))

    def _make_covariance_psd(self, cov, eps=1e-6):
        """ If a covariance matrix is not psd (numerical errors), a small value is added to its diagonal to make it psd.
        
        Parameters:
        -----------
        cov : nn.parameters.Parameter, shape [D, D]
            Covariance matrix.
        eps : float
            The small value to add to the diagonal. If it still is not psd after that, eps is multiplied by 10 and the process repeats.
        """
        # Make covariance psd in case of numerical inaccuracies
        eps = 1e-6
        while not (np.linalg.eigvals(cov.numpy()) >= 0).all():
            print(f'Covariance not positive semi-definite. Adding {eps} to the diagnoal.')
            cov += torch.eye(cov.numpy().shape[0]) * eps
            eps *= 10

    @torch.no_grad()
    def fit(self, features, labels):
        """ Fits the density models to a set of features and labels. 
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features matrix.
        labels : torch.Tensor, shape [N]
            Class labels.
        """
        if self._fitted:
            raise RuntimeError(f'GMM density model was already fitted.')
        self.components = {}
        self.coefs = nn.ParameterDict()
        self.means = nn.ParameterDict()
        self.covs = nn.ParameterDict()

        self._fit_pca(features, labels)

        for label in torch.unique(labels):
            name = f'class_{label.item()}'

            # Select all features of this label to fit a density and transform
            features_label = (features[labels == label] - self.pca_means[name]) @ self.pca_projections[name]

            # Save location and lower triangular covariance matrix to allow this module to have its state as state dict
            self.coefs[name] = nn.Parameter((labels == label).sum() / labels.size()[0], requires_grad=False)
            self.means[name] = nn.Parameter(features_label.mean(dim=0), requires_grad=False)
            self.covs[name] = nn.Parameter(cov(features_label), requires_grad=False)
            if self.diagonal_covariance:
                self.covs[name] *= torch.eye(features_label.size()[1])

            if (labels == label).sum() < features_label.size()[1]:
                warn(f'Not enough samples of class {label} ({(labels == label).sum()}) to estimate {features_label.size()[1]}-dimensional feature space covariance.')

            self._make_covariance_psd(self.covs[name], eps=1e-8)
            
        self._fitted = True
    
    @torch.no_grad()
    def forward(self, features):
        """ Gets the density at all feature points.
        
        Parameters:
        -----------
        features : torch.Tensor, shape [N, D]
            Features to get the density of.
        
        Returns:
        --------
        density : torch.Tensor, shape [N]
            Density of the GMM at each feature point.
        """
        # print(features.device)
        # for name in self.coefs:
        #     print(self.coefs[name].device, self.means[name].device, self.trils[name].device)
        if not self._fitted:
            raise RuntimeError(f'GMM density was not fitted to any data!')
        density = np.zeros(features.size(0))
        for label in self.coefs:
            features_label = (features - self.pca_means[label]) @ self.pca_projections[label]
            density_label = self.coefs[label].item() * scipy.stats.multivariate_normal.pdf(features_label.cpu().numpy(), self.means[label].cpu().numpy(), self.covs[label].cpu().numpy(), allow_singular=True)
            density_label[np.isnan(density_label)] = 0.0 # Fix some numerical issues...
            density += density_label
        return torch.tensor(density)

if __name__ == '__main__':
    x = torch.tensor([
        [1.2, 3.3],
        [0.8, 1.5],
        [-50, -51],
        [-49, -50],
        [-49, -51],
        [1.1, 0.8],
        [1.0, 1.0],
        [1.1, 1.1],
    ])
    y = torch.tensor([
        0, 0, 1, 1, 1, 0, 2, 2
    ], dtype=torch.int64)
    density = FeatureSpaceDensityGaussianPerClass()
    density.fit(x, y)
    print(density(x))   

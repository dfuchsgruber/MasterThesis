from typing import Optional
import torch
import torch.nn as nn
import pyblaze.nn as xnn
from tqdm import tqdm

class NormalizingFlow(nn.Module):
    """ Base class for fitting and evaluation of normalizing flows. 
    
    Parameters:
    -----------
    flow_type : str
        The type of flow to fit.
    num_layers : int
        How many layers to transform the density with.
    dim : int
        The dimensionality of the flow layers.
    seed : int
        Seed to seed torch with for reproducability.
    num_hidden : int
        How many hidden layers to use in the flow.
    hidden_dim : int or None
        Dimensionality of the hidden flow layers. If `None`, it is set to `dim`.
    gpu : bool
        If GPU acceleration is to be used.
    weight_decay : float
        Weight decay regularization parameter.

    """

    def __init__(self, flow_type, num_layers, dim, seed=1337, num_hidden=2, hidden_dim=None, gpu=True, weight_decay=1e-3, patience: int=5):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.num_layers = num_layers
        self.dim = dim
        self.flow_type = flow_type.lower()
        self.num_hidden = num_hidden
        if hidden_dim is None:
            hidden_dim = self.dim
        self.hidden_dim = hidden_dim
        self.gpu = gpu
        self.weight_decay = weight_decay
        self.patience = patience
            
        if self.flow_type == 'radial':
            transformations = [xnn.RadialTransform(dim) for _ in range(self.num_layers)]
        elif self.flow_type == 'maf':
            transformations = []
            for i in range(self.num_layers):
                if i > 0:
                    transformations += [xnn.FlipTransform1d()]
                transformations += [
                    xnn.MaskedAutoregressiveTransform1d(dim, *([self.hidden_dim] * self.num_hidden), constrain_scale=True),
                    xnn.BatchNormTransform1d(self.dim)
                    ]
        else:
            raise RuntimeError(f'Normalizing flow type {self.flow_type} not supported.')

        self.xflow = xnn.NormalizingFlow(transformations)

    def fit(self, x, x_val: Optional[torch.Tensor]=None, weights=None, weights_val: Optional[torch.Tensor]=None, max_iterations: int = 1000, verbose=False):
        """ 
        Fits the normalizing flow to some samples.
    
        Parameters:
        -----------
        x : torch.Tensor, shape [N, D]
            The samples to fit the flow to
        x_val : torch.Tensor, shape [N', D], optional
            Used for validation in early stopping.
        weights : torch.Tensor, shape [N] or None
            Weights for each sample in the loss likelihood loss.
            If `None`, the weight for each sample is set to 1.0
        weights_val : torch.Tensor, shape [N] or None
            Weights for each sample in the loss likelihood loss of validation samples.
            If `None`, the weight for each sample is set to 1.0
        max_iterations : int, optional
            How many iterations to perform at most.
        verbose : bool
            If progress should be printed.

        """
        if weights is None:
            weights = torch.ones(x.size(0)).float()
        if x_val is not None and weights_val is None:
            weights_val = torch.ones(x_val.size(0)).float()
        
        if torch.cuda.is_available() and self.gpu:
            if verbose:
                print(f'Fit normalizing flow with GPU acceleration.')
            x, weights = x.cuda(), weights.cuda()
            if x_val is not None:
                x_val = x_val.cuda()
                weights_val = weights_val.cuda()
            self.xflow = self.xflow.cuda()

        
        # Early stopping
        min_val_loss = float('inf')
        best_state = None
        iterations_without_improvement = 0
        stopping = False

        with torch.enable_grad():
            loss_fn = xnn.TransformedNormalLoss(reduction='none')
            optimizer = torch.optim.Adam(self.xflow.parameters(), lr=1e-3, weight_decay=self.weight_decay)
            all_iterations = range(max_iterations)
            if verbose:
                all_iterations = tqdm(all_iterations)
            for iter_ in all_iterations:
                self.xflow.train()
                optimizer.zero_grad()
                loss = loss_fn(*self.xflow(x))
                loss *= weights
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                # Validation loop in case of early stopping
                if x_val is not None:
                    with torch.no_grad():
                        self.xflow.eval()
                        loss_val = loss_fn(*self.xflow(x_val))
                        loss_val *= weights_val
                        loss_val = loss_val.mean()
                    
                    if True and iter_ % 100 == 0:
                        print(f'Iter {iter_} train loss: {loss.item():.2f}, weight {weights.sum().item():.2f} - val loss: {loss_val.item():.2f}, weight {weights_val.sum().item():.2f}')

                    if loss_val < min_val_loss:
                        iterations_without_improvement = 0
                        min_val_loss = loss_val
                        best_state = self.xflow.state_dict()
                    else:
                        iterations_without_improvement += 1
                        if iterations_without_improvement >= self.patience:
                            self.xflow.load_state_dict(best_state)
                            if verbose:
                                print(f'No improvement for {self.patience} iterations at iteration {iter_}. Stopping.')
                            stopping = True
                
                if stopping:
                    break

                # print(f'{iter} : {loss.item():.2f}')
        self.xflow.eval()
        del optimizer

    def forward(self, x):
        self.xflow.eval()
        if torch.cuda.is_available() and self.gpu:
            x = x.cuda()
            self.xflow = self.xflow.cuda()
        with torch.no_grad():
            log_density = -xnn.TransformedNormalLoss(reduction='none')(*self.xflow(x))
        return log_density.cpu()

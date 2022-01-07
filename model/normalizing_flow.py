import torch
import torch.nn as nn
import pyblaze.nn as xnn

class NormalizingFlow(nn.Module):
    """ Base class for fitting and evaluation of normalizing flows. 
    
    Parameters:
    -----------

    """

    def __init__(self, flow_type, num_layers, dim, seed=1337, num_hidden=2, hidden_dim=None):
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

    def fit(self, x, weights=None, iterations=1000):
        if weights is None:
            weights = torch.ones(x.size(0)).float()
        
        if torch.cuda.is_available() and False:
            x, weights = x.cuda(), weights.cuda()
            self.xflow = self.xflow.cuda()

        with torch.enable_grad():
            self.xflow.train()
            loss_fn = xnn.TransformedNormalLoss(reduction='none')
            optimizer = torch.optim.Adam(self.xflow.parameters(), lr=1e-3)
            for iter in range(iterations):
                optimizer.zero_grad()
                loss = loss_fn(*self.xflow(x))
                loss *= weights
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(f'{iter} : {loss.item():.2f}')

    def forward(self, x):
        self.xflow.eval()
        if torch.cuda.is_available() and False:
            x = x.cuda()
            self.xflow = self.xflow.cuda()
        with torch.no_grad():
            log_density = -xnn.TransformedNormalLoss(reduction='none')(*self.xflow(x))
        return log_density.cpu()

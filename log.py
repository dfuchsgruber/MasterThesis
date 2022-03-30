from collections import defaultdict
import pickle
import torch
import pytorch_lightning as pl
from typing import Dict, Optional
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import wandb
from PIL import Image

from plot.histogram_evolution import plot_histogram_evolution, plot_heatmap2d

class LogGradientsCallback(pl.callbacks.Callback):
    """ Callback that logs gradient information for each parameter (variance and mean). 
    
    Parameters:
    -----------
    parameters_to_log : Dict[str, str]
        A mapping from parameter name to log name for all parameters that will be logged.
    log_relative : bool, optional, default: False
        If gradient values relative to the parameter value will be logged.
    log_normalized : bool, optional, default: False
        If gradient values normalized by the parameter's norm will be logged.
    """

    def __init__(self, parameters_to_log: Dict[str, str]={}, log_relative: bool=False, log_normalized: bool=False):
        super().__init__()
        self.parameters_to_log = parameters_to_log
        self.log_relative = log_relative
        self.log_normalized = log_normalized

    def on_after_backward(self, trainer: pl.Trainer, model: pl.LightningModule):
        for name, log_name in self.parameters_to_log.items():
            named_parameters = {n : p for n, p in model.named_parameters()}
            if name in named_parameters:
                param = named_parameters[name]
                grad = param.grad
                if grad is not None:
                    # Absolute values
                    model.log(f'{log_name}_grad_mean', grad.mean())
                    model.log(f'{log_name}_grad_std', grad.std())
                    if self.log_relative:
                        # Relative to respective weight
                        model.log(f'{log_name}_grad_relative_mean', (grad / param).mean())
                        model.log(f'{log_name}_grad_relative_std', (grad / param).std())
                    if self.log_normalized:
                        # Relative to F-norm of weights
                        norm = torch.linalg.norm(param)
                        model.log(f'{log_name}_grad_normed_mean', (grad / norm).mean())
                        model.log(f'{log_name}_grad_normed_std', (grad / norm).std())

            else:
                raise RuntimeError(f'Trying to log gradient for parameter {name} but is not in named parameters, which are {list(named_parameters)}')
            
class LogWeightMatrixSpectrum(pl.callbacks.Callback):
    """ Finds all weight matrices and logs their spectrum every epoch. """

    def __init__(self, log_every_epoch=1, save_buffer_to: Optional[str]=None):
        super().__init__()
        self.log_every_epoch = log_every_epoch
        self.save_buffer_to = save_buffer_to
        self._buffer = defaultdict(list)

    def on_validation_epoch_start(self, trainer: pl.Trainer, model: pl.LightningModule):
        if trainer.global_step % self.log_every_epoch == 0:
            weights = model.get_weights()
            for name, weight in weights.items():
                u, s, v = np.linalg.svd(weight.detach().cpu().numpy(), full_matrices=False)
                self._buffer[name].append(s)
    
    def on_train_end(self, trainer, pl_module):
        if trainer.logger is not None:
            binned = {}
            heatmaps = {}
            for name, buffer in self._buffer.items():
                buffer = np.array(buffer)
                fig, ax = plot_histogram_evolution(np.array(buffer), value_name='Singular Values', time_name='Epoch', time_interval=self.log_every_epoch)
                with tempfile.NamedTemporaryFile(suffix='.png') as f:
                    fig.savefig(f.name, bbox_inches="tight")
                    with Image.open(f.name) as im:
                        binned[name] = wandb.Image(im)
                fig, ax = plot_heatmap2d(np.array(buffer), value_name='Singular Value', time_name='Epoch', time_interval=self.log_every_epoch, 
                    cbar_label='Magintude', log_scale=True)
                with tempfile.NamedTemporaryFile(suffix='.png') as f:
                    fig.savefig(f.name, bbox_inches="tight")
                    with Image.open(f.name) as im:
                        heatmaps[name] = wandb.Image(im)
            
            columns = list(binned.keys())
            trainer.logger.log_table(key='weight_matrix_spectrum', columns=columns, data=[[binned[k] for k in columns], [heatmaps[k] for k in columns]])
        if self.save_buffer_to:
            with open(self.save_buffer_to, 'wb') as f:
                pickle.dump(dict(self._buffer), f)

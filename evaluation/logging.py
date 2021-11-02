
import numpy as np
import matplotlib.pyplot as pyplot
from PIL import Image
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from warnings import warn
import wandb
import tempfile
import os.path as osp

def log_figure(logger, fig, label, save_artifact=None):
    """ Logs a figure. 
    
    Parameters:
    -----------
    logger : pytorch_lightning.loggers.Logger
        The logger to use.
    fig : plt.Figure
        The figure to log.
    label : str
        The label of the figure.
    save_artifact : path-like or None
        If a path is given, the figure is saved as pdf to that directory.
    """
    if isinstance(logger, TensorBoardLogger):
        logger.experiment.add_figure(label, fig)
    elif isinstance(logger, WandbLogger):
        with tempfile.NamedTemporaryFile(suffix='.png') as tmpfile:
            fig.savefig(tmpfile, format="png")
            logger.experiment.log({label : [wandb.Image(tmpfile.name)]})
    else:
        warn(f'Logging figures is not supported for {type(logger)}')
    if save_artifact is not None:
        fig.savefig(osp.join(save_artifact, label + '.pdf'), format='pdf')

def log_histogram(logger, values, label, **kwargs):
    """ Logs a Histogram. 
    
    Parameters:
    -----------
    logger : pytorch_lightning.loggers.Logger
        The logger to use.
    values : ndarray, shape [N]
        Values to bin.
    label : str
        The label of the histogram.
    """
    if isinstance(logger, TensorBoardLogger):
        logger.experiment.add_histogram(label, values, global_step=kwargs.get('global_step', None))
    elif isinstance(logger, WandbLogger):
        suffix = kwargs.get('label_suffix', '')
        hist = wandb.Histogram(sequence = values, num_bins = kwargs.get('num_bins', 64))
        logger.experiment.log({label + suffix : [hist]})
    else:
        warn(f'Logging histograms is not supported for {type(logger)}')

def log_embedding(logger, embeddings, label, labels=None, save_artifact=None):
    """ Logs embeddings. 
    
    Parameters:
    -----------
    logger : pytorch_lightning.loggers.Logger
        The logger to use.
    embeddings : ndarray, shape [N, D]
        Embeddings to log.
    label : str
        The label of the embedding.
    labels : ndarray, shape [N]
        Labels for each embedding point.
    save_artifact : path-like or None
        If a path is given, the embeddings is saved as npy to that directory.
    """
    if isinstance(logger, TensorBoardLogger):
        logger.experiment.add_embedding(embeddings, tag=label, metadata=labels)
    elif isinstance(logger, WandbLogger):
        table = wandb.Table(columns=list(map(str, range(embeddings.shape[1]))), data=embeddings)
        table.add_column('Labels', labels)
        logger.experiment.log({label : table})
    else:
        warn(f'Logging embeddings is not supported for {type(logger)}')
    if save_artifact is not None:
        np.save(osp.join(save_artifact, label + '.npy'), {
            'embeddings' : embeddings,
            'labels' : labels}
        )
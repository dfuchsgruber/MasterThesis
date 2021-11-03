
import numpy as np
import matplotlib.pyplot as pyplot
from PIL import Image
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from warnings import warn
import wandb
import tempfile
import os.path as osp
from itertools import chain
from collections import defaultdict

def log_metrics(logs, metrics, group, step=None):
    """ Logs metrics. 
    
    Parameters:
    -----------
    logs : dict
        A dictionary to store the logs in.
    metrics : dict
        The metrics to log.
    group : str
        In which group (table) to log the metrics in.
    """
    if step is not None:
        metrics = {f'{k}-{step}' : v for k, v in metrics.items()}
    logs[group].update(metrics)

def log_figure(logs, fig, label, group, save_artifact=None):
    """ Logs a figure. 
    
    Parameters:
    -----------
    logs : defaultdict(dict)
        A dictionary to store the logs in.
    fig : plt.Figure
        The figure to log.
    label : str
        The label of the figure.
    group : str
        In which group (table) to log it in.
    save_artifact : path-like or None
        If a path is given, the figure is saved as pdf to that directory.
    """
    if save_artifact is not None:
        fig.savefig(osp.join(save_artifact, label + '.pdf'), format='pdf')
        fig.savefig(osp.join(save_artifact, label + '.png'), format='png')
        logs[group][label] = wandb.Image(osp.join(save_artifact, label + '.png'))

def log_histogram(logs, values, label, **kwargs):
    """ Logs a Histogram. 
    
    Parameters:
    -----------
    logs : dict
        A dictionary to store the logs in.
    values : ndarray, shape [N]
        Values to bin.
    label : str
        The label of the histogram.
    """
    pass
    # suffix = kwargs.get('label_suffix', '')
    # hist = wandb.Histogram(sequence = values, num_bins = kwargs.get('num_bins', 64))
    # logs[label] = hist

def log_embedding(logs, embeddings, label, labels=None, save_artifact=None):
    """ Logs embeddings. 
    
    Parameters:
    -----------
    logs : dict
        A dictionary to store the logs in.
    embeddings : ndarray, shape [N, D]
        Embeddings to log.
    label : str
        The label of the embedding.
    labels : ndarray, shape [N]
        Labels for each embedding point.
    save_artifact : path-like or None
        If a path is given, the embeddings is saved as npy to that directory.
    """
    if save_artifact is not None:
        np.save(osp.join(save_artifact, label + '.npy'), {
            'embeddings' : embeddings,
            'labels' : labels}
        )

def build_table(logger, logs):
    """ Builds the final logging table object. 
    
    Parameters:
    -----------
    logger : pl.logging.WandbLogger
        Logger
    logs : list
        A list of dicts, each representing one log.
    """
    logs_by_group = defaultdict(list)
    for log in logs:
        for group in log:
            logs_by_group[group].append(log[group])

    for group, logs in logs_by_group.items():
        columns = sorted(list(set(chain.from_iterable(log.keys() for log in logs))))
        table = wandb.Table(columns=columns)
        for log in logs:
            data = []
            for col in columns:
                data.append(log.get(col, None))
            table.add_data(*data)
        logger.experiment.log({group : table})



def finish(logger):
    """ Ends an experiment. 
    
    Parameters:
    -----------
    logger : pytorch_lightning.loggers.Logger
        The logger to end.
    """
    if isinstance(logger, WandbLogger):
        logger.experiment.finish()
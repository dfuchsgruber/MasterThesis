
import torch
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch_geometric.loader import DataLoader
from typing import Optional, Any, Dict
import os
import logging
import numpy as np

from log import LogGradientsCallback, LogWeightMatrixSpectrum
from configuration import *
from util import suppress_stdout as context_supress_stdout
import warnings
from model_registry import ModelRegistry
import data.constants as dconstants
import model.constants as mconstants
from model.gnn import GCNLaplace, GCNLinearClassification
from model.parameterless import ParameterlessBase
import laplace

class SelfTrainingCallback(pl.callbacks.Callback):
    """ Callback that activates self-training after a certain amounts of epochs. """

    def __init__(self, num_warmup_epochs: int):
        super().__init__()
        self.num_warmup_epochs = num_warmup_epochs

    def on_epoch_start(self, trainer: pl.Trainer, model: pl.LightningModule):
        if trainer.current_epoch >= self.num_warmup_epochs:
            model.self_training = True

class SingularValueBounding(pl.callbacks.Callback):
    """ Callback that bounds the singular value after each training step. 
    
    Parameters:
    -----------
    eps : float, optional, default=0.01
        Tolerance for singular values. Will be bound to range [1/(1 + eps), 1 + eps]
    """

    def __init__(self, eps: float=0.01):
        self.eps = eps

    def on_train_batch_end(self, trainer: pl.Trainer, model: pl.LightningModule, *args, **kwargs):
        with torch.no_grad():
            for weight in model.get_weights().values():
                u, s, v = np.linalg.svd(weight.detach().cpu().numpy(), full_matrices=False)
                s = np.clip(s, 1 / (1 + self.eps), 1 + self.eps)
                weight[:, :] = torch.Tensor(u @ s @ v)

def train_pl_model(model: pl.LightningModule, config: ExperimentConfiguration, artifact_dir: str, data_loaders: Dict[str, DataLoader], 
        logger: Optional[Any]=None, callbacks: List[pl.callbacks.Callback] = []):
    """ Trains a model that can be trained with pytorch lightning. 
    
    Parameters:
    -----------
    model : pl.Module
        The model to train
    config : ExperimentConfiguration
        The configuration for the whole experiment
    artifact_dir : str
        In which directory to put artifacts.
    data_loaders : Dict[str, DataLoader]
        Data loaders used for training. See `data.constants` for keys.
    logger : Any, optional, default: None
        Logger to use for training progress.
    callbacks : List[pl.callbacks.Callback], default: []
        Callbacks that are executed on model training.
    
    Returns:
    --------
    model : pl.Module
        The input but after training.
    """
    model_registry = ModelRegistry(collection_name=config.run.model_registry_collection_name)

    checkpoint_callback = ModelCheckpoint(
        artifact_dir,
        monitor=config.training.early_stopping.monitor,
        mode=config.training.early_stopping.mode,
        save_top_k=1,
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.training.early_stopping.monitor,
        mode=config.training.early_stopping.mode,
        patience=config.training.early_stopping.patience,
        min_delta=config.training.early_stopping.min_delta,
    )
    callbacks = [cb for cb in callbacks] # Copy list
    if config.training.self_training:
        callbacks.append(SelfTrainingCallback(config.training.num_warmup_epochs))
    if config.training.singular_value_bounding:
        callbacks.append(SingularValueBounding(config.training.singular_value_bounding_eps))
    if config.logging.log_gradients:
        callbacks.append(LogGradientsCallback(config.logging.log_gradients,
            log_relative=config.logging.log_gradients_relative_to_parameter,
            log_normalized=config.logging.log_gradients_relative_to_norm)
        )
    if config.logging.log_weight_matrix_spectrum_every > 0:
        callbacks.append(LogWeightMatrixSpectrum(
            log_every_epoch=config.logging.log_weight_matrix_spectrum_every,
            save_buffer_to=config.logging.log_weight_matrix_spectrum_to_file,
        ))

    # Model training
    with context_supress_stdout(supress=config.training.suppress_stdout), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, min_epochs=config.training.min_epochs,
                            deterministic=False, callbacks=[ checkpoint_callback, early_stopping_callback, ] + callbacks,
                            logger=logger,
                            progress_bar_refresh_rate=0,
                            gpus=config.training.gpus,
                            )
        if config.run.use_pretrained_model:
            best_model_path = model_registry[config]
            if best_model_path is None:
                logging.info(f'Could not find pre-trained model.')
        else:
            best_model_path = None
        if best_model_path is None:
            trainer.fit(model, data_loaders[dconstants.TRAIN], data_loaders[dconstants.VAL])
            best_model_path = checkpoint_callback.best_model_path
            model_registry[config] = best_model_path
            os.remove(best_model_path) # Clean the checkpoint from the artifact directory
            best_model_path = model_registry[config]
        else:
            logging.info(f'Loading pre-trained model from {best_model_path}')

        model = model.load_from_checkpoint(best_model_path, strict=False, backbone_configuration=config.model)
        model.eval()
        
        # Final validation
        trainer.validate(model, data_loaders[dconstants.VAL])

        return model

def train_parameterless_model(model: ParameterlessBase, config: ExperimentConfiguration, artifact_dir: str, data_loaders: Dict[str, DataLoader], logger: Optional[Any]=None):
    """ Trains a parameterless model using its fit function.

    Parameters:
    -----------
    model : Any
        The model to train
    config : ExperimentConfiguration
        The configuration for the whole experiment
    artifact_dir : str
        In which directory to put artifacts.
    data_loaders : Dict[str, DataLoader]
        Data loaders used for training. See `data.constants` for keys.
    logger : Any, optional, default: None
        Logger to use for training progress.
    
    Returns:
    --------
    model : Any
        The input model, but after training.
    """
    data_loader_train = data_loaders[dconstants.TRAIN]
    assert len(data_loader_train) == 1
    for batch in data_loader_train:
        pass
    model.fit(batch)
    logging.info(f'Fit {model.__class__}.')
    return model

def train_laplace(model: Any, config: ExperimentConfiguration, artifact_dir: str, data_loaders: Dict[str, DataLoader], 
        logger: Optional[Any]=None, callbacks: List[pl.callbacks.Callback] = []) -> GCNLaplace:
    """ Trains a model if neccessary and applies laplace approximation to a linear output layer.
    
    Parameters:
    -----------
    model : Any
        The model to train
    config : ExperimentConfiguration
        The configuration for the whole experiment
    artifact_dir : str
        In which directory to put artifacts.
    data_loaders : Dict[str, DataLoader]
        Data loaders used for training. See `data.constants` for keys.
    logger : Any, optional, default: None
        Logger to use for training progress.
    callbacks : List[pl.callbacks.Callback], default: []
        Callbacks that are executed on model training.
    
    Returns:
    --------
    model : Any
        The input model, but after training.
    """
    assert isinstance(model.backbone, GCNLinearClassification), f'Laplace posterior can only be fit to linear classification architectures, not to {type(model.backbone)}'
    encoder: GCNLinearClassification = train_pl_model(model, config, artifact_dir, data_loaders, logger=logger, callbacks=callbacks).backbone
    if config.training.gpus:
        encoder = encoder.cuda()
    logging.info('Decomposing classifier')
    # Construct a simple tensor dataset x, y
    data_train = data_loaders[dconstants.TRAIN].dataset[0]
    data_val = data_loaders[dconstants.VAL].dataset[0]
    encoder.eval()
    encoder_devices = list(set(p.device for p in encoder.parameters()))
    if len(encoder_devices) != 1:
        raise RuntimeError(f'Multiple encoder devices found: {encoder_devices}')
    device = encoder_devices[0]
    encoder = encoder.to(device)
    logging.info(f'Using device {device}')
    with torch.no_grad():
        h_train = encoder(data_train.to(device), sample=False).get_features(layer=-2).to(device)
        h_val = encoder(data_val.to(device), sample=False).get_features(layer=-2).to(device)

    laplace_data_train = TensorDataset(
        h_train[data_train.mask], data_train.y[data_train.mask]
    )
    laplace_data_val = TensorDataset(
        h_val[data_val.mask], data_val.y[data_val.mask]
    )
    if not config.model.laplace.gpu:
        encoder = encoder.cpu()
        laplace_data_train.tensors = tuple(t.to('cpu') for t in laplace_data_train.tensors)
        laplace_data_val.tensors = tuple(t.to('cpu') for t in laplace_data_val.tensors)

    if config.model.laplace.batch_size > 0:
        batch_size_train, batch_size_val = config.model.laplace.batch_size, config.model.laplace.batch_size
    else:
        batch_size_train, batch_size_val = len(laplace_data_train), len(laplace_data_val)

    laplace_loader_train = DataLoader(laplace_data_train, batch_size=batch_size_train, shuffle=False)
    laplace_loader_val = DataLoader(laplace_data_val, batch_size=batch_size_val, shuffle=False)
    head = encoder.head.conv.linear



    la = laplace.Laplace(
        head,
        'classification',
        subset_of_weights='all',
        hessian_structure=config.model.laplace.hessian_structure,
    )
    pl.seed_everything(config.model.laplace.seed) # To ensure reproducability
    logging.info('Fitting laplace posterior')
    la.fit(laplace_loader_train)
    logging.info('Optimizing prior precision')
    la.optimize_prior_precision(method='CV', val_loader=laplace_loader_val)

    model.backbone = GCNLaplace(encoder.convs, la, config.model)
    return model.cpu()

def train_model(model: Any, config: ExperimentConfiguration, artifact_dir: str, data_loaders: Dict[str, DataLoader], logger: Optional[Any]=None,
    callbacks: List[pl.callbacks.Callback] = []) -> Any:
    """ Trains a model if neccessary.
    
    Parameters:
    -----------
    model : Any
        The model to train
    config : ExperimentConfiguration
        The configuration for the whole experiment
    artifact_dir : str
        In which directory to put artifacts.
    data_loaders : Dict[str, DataLoader]
        Data loaders used for training. See `data.constants` for keys.
    logger : Any, optional, default: None
        Logger to use for training progress.
    return_trainer : bool, optional, default: False
        If the trainer should also be returned.
    callbacks : List[pl.callbacks.Callback], default: []
        Callbacks that are executed on model training.
    
    Returns:
    --------
    model : Any
        The input model, but after training.
    """
    if model.training_type == mconstants.TRAIN_PL:
        return train_pl_model(model, config, artifact_dir, data_loaders, logger=logger, callbacks=callbacks)
    elif model.training_type == mconstants.TRAIN_PARAMETERLESS:
        return train_parameterless_model(model, config, artifact_dir, data_loaders, logger=logger)
    elif model.training_type == mconstants.TRAIN_LAPLACE:
        return train_laplace(model, config, artifact_dir, data_loaders, logger=logger, callbacks=callbacks)
    else:
        raise ValueError(f'Training for type {model.training_type} not implemented.')
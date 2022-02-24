
import torch
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch_geometric.loader import DataLoader
from typing import Optional, Any, Dict
import os
import logging

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


def train_pl_model(model: pl.LightningModule, config: ExperimentConfiguration, artifact_dir: str, data_loaders: Dict[str, DataLoader], logger: Optional[Any]=None):
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
    callbacks = []
    if config.training.self_training:
        callbacks.append(SelfTrainingCallback(config.training.num_warmup_epochs))

    # Model training
    with context_supress_stdout(supress=config.training.suppress_stdout), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, deterministic=True, callbacks=[ checkpoint_callback, early_stopping_callback, ] + callbacks,
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
        trainer.validate(model, data_loaders[dconstants.VAL], ckpt_path=best_model_path)

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

def train_laplace(model: Any, config: ExperimentConfiguration, artifact_dir: str, data_loaders: Dict[str, DataLoader], logger: Optional[Any]=None) -> GCNLaplace:
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
    
    Returns:
    --------
    model : Any
        The input model, but after training.
    """
    assert isinstance(model.backbone, GCNLinearClassification), f'Laplace posterior can only be fit to Linear classification architectures, not to {type(model.backbone)}'
    encoder: GCNLinearClassification = train_pl_model(model, config, artifact_dir, data_loaders, logger=logger).backbone
    logging.info('Decomposing classifier')
    # Construct a simple tensor dataset x, y
    data_train = data_loaders[dconstants.TRAIN].dataset[0]
    data_val = data_loaders[dconstants.VAL].dataset[0]
    encoder.eval()
    with torch.no_grad():
        h_train = encoder(data_train, sample=False).get_features(layer=-2)
        h_val = encoder(data_val, sample=False).get_features(layer=-2)

    laplace_data_train = TensorDataset(
        h_train[data_train.mask], data_train.y[data_train.mask]
    )
    laplace_data_val = TensorDataset(
        h_val[data_val.mask], data_val.y[data_val.mask]
    )
    laplace_loader_train = DataLoader(laplace_data_train, batch_size=len(laplace_data_train), shuffle=False)
    laplace_loader_val = DataLoader(laplace_data_val, batch_size=len(laplace_data_val), shuffle=False)
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
    return model

def train_model(model: Any, config: ExperimentConfiguration, artifact_dir: str, data_loaders: Dict[str, DataLoader], logger: Optional[Any]=None) -> Any:
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
    
    Returns:
    --------
    model : Any
        The input model, but after training.
    """
    if model.training_type == mconstants.TRAIN_PL:
        return train_pl_model(model, config, artifact_dir, data_loaders, logger=logger)
    elif model.training_type == mconstants.TRAIN_PARAMETERLESS:
        return train_parameterless_model(model, config, artifact_dir, data_loaders, logger=logger)
    elif model.training_type == mconstants.TRAIN_LAPLACE:
        return train_laplace(model, config, artifact_dir, data_loaders, logger=logger)
    else:
        raise ValueError(f'Training for type {model.training_type} not implemented.')
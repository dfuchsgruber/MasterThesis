
from audioop import add
from copy import deepcopy
from model.semi_supervised_node_classification import SemiSupervisedNodeClassification
import torch
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch_geometric.loader import DataLoader
from typing import Optional, Any, Dict, Tuple
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

class Callbacks:
    """ Class to manage all callbacks used during training. """

    def __init__(self, config: ExperimentConfiguration, artifact_dir: str, 
            additional_callbacks: Union[Dict[str, pl.callbacks.Callback], List[pl.callbacks.Callback]] = {}):

        self._artifact_dir = artifact_dir
        self.reset_early_stopping(config.training.early_stopping)
        self.reset_checkpoint(config.training.early_stopping)

        if config.training.self_training:
            self.self_training = (SelfTrainingCallback(config.training.num_warmup_epochs))
        else:
            self.self_training = None
        if config.training.singular_value_bounding:
            self.singular_value_bounding = (SingularValueBounding(config.training.singular_value_bounding_eps))
        else:
            self.singular_value_bounding = None
        if config.logging.log_gradients:
            self.log_gradients = LogGradientsCallback(config.logging.log_gradients,
                log_relative=config.logging.log_gradients_relative_to_parameter,
                log_normalized=config.logging.log_gradients_relative_to_norm)
        else:
            self.log_gradients = None
        if config.logging.log_weight_matrix_spectrum_every > 0:
            self.log_weight_matrix_spectrum = LogWeightMatrixSpectrum(
                log_every_epoch=config.logging.log_weight_matrix_spectrum_every,
                save_buffer_to=config.logging.log_weight_matrix_spectrum_to_file,
            )
        else:
            self.log_weight_matrix_spectrum = None

        self.additional_callbacks = {}

        if isinstance(additional_callbacks, list):
            for cb in additional_callbacks:
                self.add(cb)
        elif isinstance(additional_callbacks, Mapping):
            for name, cb in additional_callbacks.items():
                self.add(cb, name=name)
        else:
            raise RuntimeError(f'Callback buffer is to be initialized with unsupported datatype {type(additional_callbacks)}')

    def reset_early_stopping(self, config: EarlyStoppingConfiguration):
        """ Sets up a new early stopping callback. """
        self.early_stopping = EarlyStopping(
            monitor=config.monitor,
            mode=config.mode,
            patience=config.patience,
            min_delta=config.min_delta,
            )

    def reset_checkpoint(self, config: EarlyStoppingConfiguration):
        """ Sets up a new checkpoint callback. """ 
        self.checkpoint = ModelCheckpoint(
            self._artifact_dir,
            monitor=config.monitor,
            mode=config.mode,
            save_top_k=1,
            )

    def add(self, callback: pl.callbacks.Callback, name: Optional[str] = None):
        """ Adds a callback to the buffer. """
        if name is None:
            name = f'callback_{len(self.additional_callbacks)}'
            self.additional_callbacks[name] = callback

    def to_list(self) -> List[pl.callbacks.Callback]:
        """ Returns all callbacks as list. """
        cbs = [self.checkpoint, self.early_stopping, self.self_training, self.singular_value_bounding, self.log_gradients, self.log_weight_matrix_spectrum]
        cbs += list(self.additional_callbacks.values())
        return [cb for cb in cbs if cb is not None]

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

def finetune_pl_model(model: SemiSupervisedNodeClassification, config: ExperimentConfiguration, logger, callback_buffer: Callbacks, 
        data_loaders: Dict[str, DataLoader]) -> Tuple[str, pl.Trainer]:
    """ Performs finetuning on a trained model. 
    
    Parameters:
    -----------
    model : pl.Module
        The model to train
    config : ExperimentConfiguration
        The configuration for the whole experiment
    logger : Any
        The logger to log the finetuning with.
    callback_buffer : Callbacks
        A buffer for all callbacks. Early stopping and checkpointing will be reset.
    data_loaders : Dict[str, DataLoader]
        Data loaders used for training. See `data.constants` for keys.
    
    Returns:
    --------
    path : str
        Path to the model checkpoint.
    trainer : pl.Trainer
        The trainer used for finetuning.
    """
    # Reset callbacks for early stopping and checkpointing
    callback_buffer.reset_early_stopping(config.training.early_stopping)
    callback_buffer.reset_checkpoint(config.training.early_stopping)

    # Create a trainer
    trainer = pl.Trainer(max_epochs=config.training.finetuning.max_epochs,
                            deterministic=False, callbacks = callback_buffer.to_list(),
                            logger=logger,
                            progress_bar_refresh_rate=0,
                            gpus=config.training.gpus,
                            )
    # Finetuning
    if config.training.finetuning.reconstruction is not None:
        model.add_reconstruction_loss(config.training.finetuning.reconstruction)
        logging.info(f'Added reconstruction loss.')
    if config.training.finetuning.feature_reconstruction is not None:
        model.add_feature_reconstruction(
            data_loaders[dconstants.TRAIN].dataset[0].x.size(1),
            config.training.finetuning.feature_reconstruction,
        )
        logging.info(f'Added feature reconstruction loss.')
    trainer.fit(model, data_loaders[dconstants.TRAIN], data_loaders[dconstants.VAL])
    return callback_buffer.checkpoint.best_model_path, trainer

def _train_pl_model_impl(model: SemiSupervisedNodeClassification, config: ExperimentConfiguration, 
    trainer: pl.Trainer, logger, callback_buffer: Callbacks, data_loaders: Dict[str, DataLoader]) -> Tuple[str, pl.Trainer]:
    """ Performs model training with given trainer or loads the model from the registry. 
    In the case of finetuning a modelo, this function recurses to train the base model.

    Parameters:
    -----------
    model : SemiSupervisedNodeClassification
        The model to train.
    config : ExperimentConfiguration
        The configuration associated with the experiment.
    trainer : pl.Trainer
        The trainer instance to train with.
    logger : Any
        The logger.
    callback_buffer : Callbacks
        The callbacks that are executed.
    data_loaders : Dict[str, DataLoader]
        Data loaders to train on.
    
    Returns:
    --------
    path : str
        Path to the trained model checkpoint.
    trainer : pl.Trainer
        The (last) trainer used for training. In the case of finetuning, this might be the finetuning trainer
        (only if the model was not loaded from the model registry storage).
    """ 
    model_registry = ModelRegistry(collection_name=config.run.model_registry_collection_name)
    
    # Try finding a pretrained checkpoint
    if config.run.use_pretrained_model:
        best_model_path = model_registry[config]
        if best_model_path is None:
            logging.info(f'Could not find pre-trained model.')
    else:
        best_model_path = None

    if best_model_path is None:
        # Train the model with a given trainer
        if config.training.finetuning.enable:

            # (Recursively) get a base model
            base_config = deepcopy(config)
            base_config.training.finetuning = FinetuningConfiguration(enable=False)
            logging.info(f'Training / loading base model...')
            best_model_path, _ = _train_pl_model_impl(model, base_config, trainer, logger, callback_buffer, data_loaders)
            logging.info(f'Loading base model from {best_model_path}')
            model.load_from_checkpoint(best_model_path, strict=False, backbone_configuration=base_config.model)

            # Finetune base model
            best_model_path, trainer = finetune_pl_model(model, config, logger, callback_buffer, data_loaders)
        else:
            logging.info('Starting model training.')
            # Train a model
            trainer.fit(model, data_loaders[dconstants.TRAIN], data_loaders[dconstants.VAL])
            best_model_path = callback_buffer.checkpoint.best_model_path

        # Model was either trained or finetuned, but in any case `config` was not present in the registry
        model_registry[config] = best_model_path # Copies the checkpoint to the registry storage directory
        os.remove(best_model_path) # Clean the checkpoint from the artifact directory
        best_model_path = model_registry[config] # Gets the path in the registroy storage directory

    else:
        logging.info(f'Found pre-trained model at {best_model_path}')

    return best_model_path, trainer

def train_pl_model(model: pl.LightningModule, config: ExperimentConfiguration, artifact_dir: str, data_loaders: Dict[str, DataLoader], 
        logger: Optional[Any]=None, callbacks: Union[List[pl.callbacks.Callback], Dict[str, pl.callbacks.Callback]] = []):
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
    callbacks : Union[List[pl.callbacks.Callback], Dict[str, pl.callbacks.Callback]], default: []
        Callbacks that are executed on model training.
    
    Returns:
    --------
    model : pl.Module
        The input but after training.
    """
    callback_buffer = Callbacks(config, artifact_dir, additional_callbacks=callbacks)

    with context_supress_stdout(supress=config.training.suppress_stdout), warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        trainer = pl.Trainer(max_epochs=config.training.max_epochs, min_epochs=config.training.min_epochs,
                            deterministic=False, callbacks = callback_buffer.to_list(),
                            logger=logger,
                            progress_bar_refresh_rate=0,
                            gpus=config.training.gpus,
                            )
        best_model_path, trainer = _train_pl_model_impl(model, config, trainer, logger, callback_buffer, data_loaders)
        logging.info(f'Loading model for evaluation from {best_model_path}.')
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
from typing import Union
from sacred import Experiment
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import seml
import os.path as osp
import os
import json
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import attr
import logging as l

l.basicConfig(level=l.INFO)

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from util import format_name
from util import suppress_stdout as context_supress_stdout
from model.semi_supervised_node_classification import SemiSupervisedNodeClassification, Ensemble
from data.util import data_get_num_attributes, data_get_num_classes
from data.construct import load_data_from_configuration
import data.constants as dconstants
from evaluation.pipeline import Pipeline
from evaluation.logging import finish as finish_logging, build_table
from model_registry import ModelRegistry
import configuration
import seed

os.environ['WANDB_START_METHOD'] = 'thread'

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

class SelfTrainingCallback(pl.callbacks.Callback):
    """ Callback that activates self-training after a certain amounts of epochs. """

    def __init__(self, num_warmup_epochs: int):
        super().__init__()
        self.num_warmup_epochs = num_warmup_epochs

    def on_epoch_start(self, trainer: pl.Trainer, model: pl.LightningModule):
        if trainer.current_epoch >= self.num_warmup_epochs:
            model.self_training = True

class ExperimentWrapper:

    def __init__(self, init_all=True, collection_name=None, run_id=None):
        if init_all:
            pass
        self.collection_name = collection_name
        self.run_id = run_id

    @ex.capture()
    def train(
        self, 
        model: Union[dict, configuration.ModelConfiguration] = {}, 
        data: Union[dict, configuration.DataConfiguration] = {}, 
        evaluation: Union[dict, configuration.EvaluationConfiguration] = {}, 
        training: Union[dict, configuration.TrainingConfiguration] = {}, 
        run: Union[dict, configuration.RunConfiguration] = {}, 
        ensemble: Union[dict, configuration.EnsembleConfiguration] = {}, 
        logging: Union[dict, configuration.LoggingConfiguration] = {}
        ):
        config = configuration.ExperimentConfiguration(data=data, model=model, evaluation=evaluation, training=training, run=run, ensemble = ensemble, logging=logging)

        if not torch.cuda.is_available() and config.training.gpus > 0:
            l.warn(f'Requested {config.training.gpus} GPU devices but none are available. Not using GPU.')
            config.training.gpus = 0

        l.info(f'Logging to collection {self.collection_name}')

        model_registry = ModelRegistry(collection_name=config.run.model_registry_collection_name)

        data_split_seed = seed.data_split_seeds()[config.run.split_idx]
        model_seed_generator = iter(seed.SeedIterator(seed.model_seeds()[config.run.initialization_idx]))
        data_dict, fixed_vertices = load_data_from_configuration(config.data, data_split_seed)

        run_name = format_name(config.run.name, config.run.args, attr.asdict(config))
        # One global logger for all splits and initializations
        logger = WandbLogger(save_dir=config.logging.logging_dir, project=str(self.collection_name), name=f'{run_name}')
        logger.log_hyperparams(attr.asdict(config))

        run_artifact_dir = artifact_dir = osp.join(config.logging.artifact_dir, str(self.collection_name), f'{run_name}')
        all_logs = [] # Logs from each run
        all_artifacts = [] # Paths to all artifacts generated during evaluation

        # Build evaluation pipeline
        pipeline = Pipeline(config.evaluation.pipeline, config.evaluation, gpus=config.training.gpus, 
            ignore_exceptions=config.evaluation.ignore_exceptions)

        # Iterating over all dataset splits
        result = defaultdict(list)

        data_loaders = {
            name : DataLoader(data, batch_size=1, shuffle=False) for name, data in data_dict.items()
        }
        config.registry.split_seed = data_split_seed

        ensembles = []
        for ensemble_idx in range(config.ensemble.num_members):
            model_seed = next(model_seed_generator)

            config.registry.model_seed = model_seed

            pl.seed_everything(model_seed)
            model = SemiSupervisedNodeClassification(
                config.model, 
                data_get_num_attributes(data_dict[dconstants.TRAIN][0]), 
                data_get_num_classes(data_dict[dconstants.TRAIN][0]), 
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
            # print(f'Model parameters (trainable / all): {module_numel(model, only_trainable=True)} / {module_numel(model, only_trainable=False)}')
            
            artifact_dir = osp.join(run_artifact_dir, f'{config.run.split_idx}-{config.run.initialization_idx}-{ensemble_idx}')
            os.makedirs(artifact_dir, exist_ok=True)

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
                        l.info(f'Could not find pre-trained model.')
                else:
                    best_model_path = None
                if best_model_path is None:
                    trainer.fit(model, data_loaders[dconstants.TRAIN], data_loaders[dconstants.VAL])
                    best_model_path = checkpoint_callback.best_model_path
                    model_registry[config] = best_model_path
                    os.remove(best_model_path) # Clean the checkpoint from the artifact directory
                    best_model_path = model_registry[config]
                else:
                    l.info(f'Loading pre-trained model from {best_model_path}')

                model = model.load_from_checkpoint(best_model_path, strict=False, backbone_configuration=config.model)
                model.eval()
                
            # Add the model to the ensemble
            ensembles.append(model.eval())

        # An ensemble of 1 model behaves just like one model of this type: 
        # Therefore we always deal with an "ensemble", even if there is only one member
        model = Ensemble(ensembles, config.ensemble.num_samples)
        model.clear_and_disable_cache()
        val_metrics = {
            name : trainer.validate(model, data_loaders[name], ckpt_path=best_model_path)
            for name in (dconstants.VAL, )
        }
        for name, metrics in val_metrics.items():
            for val_metric in metrics:
                for metric, value in val_metric.items():
                    result[f'{metric}-{name}-{ensemble_idx}'].append(value)

        # Run evaluation pipeline
        l.info(f'Executing pipeline...')
        if config.evaluation.print_pipeline:
            l.info(str(pipeline))
        logs = defaultdict(dict)

        pipeline_metrics = {} # Metrics logged by the pipeline
        pipeline(
            model=model,
            data_loaders = data_loaders,
            logger=logger,
            config=config,
            artifact_directory=artifact_dir,
            logs=logs,
            metrics=pipeline_metrics,
            ensemble_members=ensembles,
            artifacts=all_artifacts,
        )

        all_logs.append(logs)
        for metric, value in pipeline_metrics.items():
            result[metric].append(value)
        
        plt.close('all')

        # Build wandb table for everything that was logged by the pipeline
        build_table(logger, all_logs)
        os.makedirs(run_artifact_dir, exist_ok=True)
        metrics_path = osp.join(run_artifact_dir, 'metrics.json')
        with open(metrics_path, 'w+') as f:
            json.dump({metric : values for metric, values in result.items()}, f)
        finish_logging(logger)

        # Remove artifacts if they were not to be logged
        if not config.evaluation.save_artifacts:
            l.info(f'Deleting pipeline artifacts...')
            for path in all_artifacts:
                os.remove(path)

        return {'results' : {metric : values for metric, values in result.items()}, 'configuration' : attr.asdict(config)}


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(_config, experiment=None,):
    run_id = _config['overwrite']
    db_collection = _config['db_collection']
    if experiment is None:
        experiment = ExperimentWrapper(collection_name=db_collection, run_id=run_id)
    return experiment.train()

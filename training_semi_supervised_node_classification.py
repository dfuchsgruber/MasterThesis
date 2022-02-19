from typing import Union
from sacred import Experiment
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import seml
import os.path as osp
import os
import json
import attr
from collections import defaultdict
import matplotlib.pyplot as plt
import logging as l

l.basicConfig(level=l.INFO)

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from util import format_name
from model.semi_supervised_node_classification import Ensemble
from data.util import data_get_num_attributes, data_get_num_classes
from data.construct import load_data_from_configuration
import data.constants as dconstants
from evaluation.pipeline import Pipeline
from evaluation.logging import finish as finish_logging, build_table
import configuration
import seed
from train import train_model
from model.build import make_model

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
        if config.run.use_default_configuration:
            configuration.update_with_default_configuration(config)

        # Discard demanded GPUs if None available
        if not torch.cuda.is_available() and config.training.gpus > 0:
            l.warn(f'Requested {config.training.gpus} GPU devices but none are available. Not using GPU.')
            config.training.gpus = 0
        l.info(f'Logging to collection {self.collection_name}')

        # Data Loading
        data_split_seed = seed.data_split_seeds()[config.run.split_idx]
        config.registry.split_seed = data_split_seed
        model_seed_generator = iter(seed.SeedIterator(seed.model_seeds()[config.run.initialization_idx]))
        data_dict, fixed_vertices = load_data_from_configuration(config.data, data_split_seed)
        data_loaders = {
            name : DataLoader(data, batch_size=1, shuffle=False) for name, data in data_dict.items()
        }

        # Setup logging
        run_name = format_name(config.run.name, config.run.args, attr.asdict(config))
        logger = WandbLogger(save_dir=config.logging.logging_dir, project=str(self.collection_name), name=f'{run_name}')
        logger.log_hyperparams(attr.asdict(config))
        run_artifact_dir = artifact_dir = osp.join(config.logging.artifact_dir, str(self.collection_name), f'{run_name}')
        all_logs = [] # Logs from each run
        all_artifacts = [] # Paths to all artifacts generated during evaluation

        # Setup evaluation pipeline
        pipeline = Pipeline(config.evaluation.pipeline, config.evaluation, gpus=config.training.gpus, 
            ignore_exceptions=config.evaluation.ignore_exceptions)

        # Model loading and training
        result = defaultdict(list)
        ensembles = []
        for ensemble_idx in range(config.ensemble.num_members):

            artifact_dir = osp.join(run_artifact_dir, f'{config.run.split_idx}-{config.run.initialization_idx}-{ensemble_idx}')
            os.makedirs(artifact_dir, exist_ok=True)
            model_seed = next(model_seed_generator)
            config.registry.model_seed = model_seed
            pl.seed_everything(model_seed)
            model = make_model(config, data_get_num_attributes(data_dict[dconstants.TRAIN][0]), 
                data_get_num_classes(data_dict[dconstants.TRAIN][0])) 
            model = train_model(model, config, artifact_dir, data_loaders, logger=logger).eval()
            ensembles.append(model)

        # An ensemble of 1 model behaves just like one model of this type: 
        # Therefore we always deal with an "ensemble", even if there is only one member
        model = Ensemble(ensembles, config.ensemble.num_samples, sample_at_eval=config.evaluation.sample)
        model.clear_and_disable_cache()

        # Run evaluation pipeline
        l.info(f'Executing pipeline...')
        if config.evaluation.print_pipeline:
            l.info(str(pipeline))
        logs = defaultdict(dict)

        pipeline_metrics = {} # Metrics logged by the pipeline
        pipeline(
            model=model.eval(),
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
        
        plt.close('all') # To not run into OOM or plt warnings

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

@ex.command(unobserved=True)
def get_experiment(init_all=False):
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment

@ex.automain
def train(_config, experiment=None,):
    run_id = _config['overwrite']
    db_collection = _config['db_collection']
    if experiment is None:
        experiment = ExperimentWrapper(collection_name=db_collection, run_id=run_id)
    return experiment.train()

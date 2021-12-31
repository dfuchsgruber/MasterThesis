from sacred import Experiment
import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import seml
import os.path as osp
import os
import json
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import configuration

from util import format_name
from util import suppress_stdout as context_supress_stdout
from model.util import module_numel
from model.semi_supervised_node_classification import SemiSupervisedNodeClassification, Ensemble
from data.gust_dataset import GustDataset
from data.util import data_get_num_attributes, data_get_num_classes
from data.construct import load_data_from_configuration
import data.constants as data_constants
from seed import model_seeds
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from evaluation.pipeline import Pipeline
from evaluation.logging import finish as finish_logging, build_table
from model_registry import ModelRegistry

NAME_SPLIT, NAME_INIT = 'split', 'init'
SPLIT_INIT_GROUP = 'split_and_init'

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

    def __init__(self, init_all=True, collection_name=None, run_id=None, model_registry=None):
        if init_all:
            self.init_all()
        self.collection_name = collection_name
        self.run_id = run_id
        if model_registry is None:
            model_registry = ModelRegistry()
        self.model_registry = model_registry

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @ex.capture(prefix="data")
    def init_dataset(self, dataset, num_dataset_splits, train_portion, val_portion, test_portion, test_portion_fixed,
                        train_labels='all', train_labels_remove_other=False,
                        val_labels='all', val_labels_remove_other=False,
                        split_type='stratified', base_labels='all', 
                        type='gust', corpus_labels='all',
                        min_token_frequency=10,
                        preprocessing='bag_of_words',
                        language_model='bert-base-uncased',
                        drop_train_vertices_portion = 0.0,
                        ):
        self.data_config = {
            'dataset' : dataset,
            'num_dataset_splits' : num_dataset_splits,
            'train_portion' : train_portion,
            'val_portion' : val_portion,
            'test_portion' : test_portion,
            'test_portion_fixed' : test_portion_fixed,
            'train_labels' : train_labels,
            'train_labels_remove_other' : train_labels_remove_other,
            # 'train_labels_compress' : train_labels_compress, # Train labels should always be compressed
            'val_labels' : val_labels,
            'val_labels_remove_other' : val_labels_remove_other,
            # 'val_labels_compress' : val_labels_compress, # Val labels should never be compressed
            'split_type' : split_type,
            'base_labels' : base_labels,
            'type' : type,
            'corpus_labels' : corpus_labels,
            'min_token_frequency' : min_token_frequency,
            'preprocessing' : preprocessing,
            'language_model' : language_model,
            'drop_train_vertices_portion' : drop_train_vertices_portion,

        }
        # self.data_mask_split, self.data_mask_test_fixed = stratified_split_with_fixed_test_set_portion(self.data[0].y.numpy(), num_dataset_splits, 
        #     portion_train=train_portion, portion_val=val_portion, portion_test_fixed=test_portion_fixed, portion_test_not_fixed=test_portion)

    @ex.capture(prefix="model")
    def init_model(self, model_type: str, hidden_sizes: list, weight_scale: float, num_initializations: int, use_spectral_norm: bool, num_heads=-1, 
        diffusion_iterations=5, teleportation_probability=0.1, use_bias=True, activation='leaky_relu', leaky_relu_slope=0.01, normalize=True,
        residual=False, freeze_residual_projection=False, num_ensemble_members=1, num_samples=1, dropout=0.0, drop_edge=0.0, use_spectral_norm_on_last_layer=True,
        self_loop_fill_value=1.0):
        self.model_config = {
            'hidden_sizes' : hidden_sizes,
            'weight_scale' : weight_scale,
            'use_spectral_norm' : use_spectral_norm,
            'num_heads' : num_heads,
            'diffusion_iterations' : diffusion_iterations,
            'teleportation_probability' : teleportation_probability,
            'model_type' : model_type,
            'use_bias' : use_bias,
            'activation' : activation,
            'leaky_relu_slope' : leaky_relu_slope,
            'normalize' : normalize,
            'residual' : residual,
            'freeze_residual_projection' : freeze_residual_projection,
            'num_ensemble_members' : num_ensemble_members,
            'num_samples' : num_samples,
            'dropout' : dropout,
            'drop_edge' : drop_edge,
            'use_spectral_norm_on_last_layer' : use_spectral_norm_on_last_layer,
            'self_loop_fill_value' : self_loop_fill_value,
        }
        self.model_seeds = model_seeds(num_initializations, model_name=model_type)

    def init_all(self):
        """ Sequentially run the sub-initializers of the experiment. """
        self.init_dataset()
        self.init_model()
        self.init_evaluation()
        self.init_run()

    @ex.capture(prefix='evaluation')
    def init_evaluation(self, pipeline=[], print_pipeline=False, ignore_exceptions=True, log_plots=True, save_artifacts=False):
        self.evaluation_config = {
            'pipeline' : pipeline,
            'print_pipeline' : print_pipeline,
            'ignore_exceptions' : ignore_exceptions,
            'log_plots' : log_plots,
            'save_artifacts' : save_artifacts,
        }

    @ex.capture(prefix='run')
    def init_run(self, name='', args=[]):
        self.run_name_format = name
        self.run_name_format_args = args

    @ex.capture(prefix="training")
    def train(self, max_epochs, learning_rate, early_stopping, gpus, suppress_stdout=True):
        
        training_config = {
            'max_epochs' : max_epochs,
            'learning_rate' : learning_rate,
            'early_stopping' : early_stopping,
            'gpus' : gpus,
        }
        # Setup config and name of the run(s)
        config = configuration.get_experiment_configuration({
            'model' : self.model_config,
            'data' : self.data_config,
            'evaluation' : self.evaluation_config,
            'training' : training_config,
        })

        # Data loading
        data_list, dataset_fixed = load_data_from_configuration(config['data'])
        
        run_name = format_name(self.run_name_format, self.run_name_format_args, config)
        # One global logger for all splits and initializations
        logger = WandbLogger(save_dir=osp.join('/nfs/students/fuchsgru/wandb'), project=str(self.collection_name), name=f'{run_name}')
        logger.log_hyperparams(config)

        run_artifact_dir = artifact_dir = osp.join('/nfs/students/fuchsgru/artifacts', str(self.collection_name), f'{run_name}')
        all_logs = [] # Logs from each run
        all_artifacts = [] # Paths to all artifacts generated during evaluation

        # Iterating over all dataset splits
        result = defaultdict(list)
        for split_idx, data_dict in enumerate(data_list):

            data_loaders = {
                name : DataLoader(data, batch_size=1, shuffle=False) for name, data in data_dict.items()
            }

            # Re-initializing the model multiple times to average over results
            for reinitialization, seed in enumerate(self.model_seeds):
                pl.seed_everything(seed)
                ensembles = []
                for ensemble_idx in range(config['model']['num_ensemble_members']):

                    model = SemiSupervisedNodeClassification(
                        config['model'], 
                        data_get_num_attributes(data_dict[data_constants.TRAIN][0]), 
                        data_get_num_classes(data_dict[data_constants.TRAIN][0]), 
                        learning_rate=config['training']['learning_rate'],
                        self_loop_fill_value=config['model']['self_loop_fill_value'],
                    )
                    # print(f'Model parameters (trainable / all): {module_numel(model, only_trainable=True)} / {module_numel(model, only_trainable=False)}')


                    artifact_dir = osp.join(run_artifact_dir, f'{split_idx}-{reinitialization}-{ensemble_idx}')
                    os.makedirs(artifact_dir, exist_ok=True)

                    checkpoint_callback = ModelCheckpoint(
                        artifact_dir,
                        monitor=config['training']['early_stopping']['monitor'],
                        mode=config['training']['early_stopping']['mode'],
                        save_top_k=1,
                    )
                    early_stopping_callback = EarlyStopping(
                        monitor=config['training']['early_stopping']['monitor'],
                        mode=config['training']['early_stopping']['mode'],
                        patience=config['training']['early_stopping']['patience'],
                        min_delta=config['training']['early_stopping']['min_delta'],
                    )

                    # Model training
                    with context_supress_stdout(supress=suppress_stdout), warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        trainer = pl.Trainer(max_epochs=config['training']['max_epochs'], deterministic=True, callbacks=[ checkpoint_callback, early_stopping_callback, ],
                                            logger=logger,
                                            progress_bar_refresh_rate=0,
                                            gpus=config['training']['gpus'],
                                            )
                        registry_config = config | {
                            'split_idx' : split_idx,
                            'model_seed' : int(seed),
                            'ensemble_idx' : ensemble_idx,
                            'reinitialization' : reinitialization,
                        }
                        best_model_path = self.model_registry[registry_config]
                        if best_model_path is None:
                            trainer.fit(model, data_loaders[data_constants.TRAIN], data_loaders[data_constants.VAL_REDUCED])
                            best_model_path = checkpoint_callback.best_model_path
                            self.model_registry[registry_config] = best_model_path
                            os.remove(best_model_path) # Clean the checkpoint from the artifact directory
                            best_model_path = self.model_registry[registry_config]
                        else:
                            print(f'Loading pre-trained model from {best_model_path}')

                        model = model.load_from_checkpoint(best_model_path)
                        model.eval()
                        

                    # Add the model to the ensemble
                    ensembles.append(model.eval())

                model = Ensemble(ensembles, self.model_config.get('num_samples', 1)) # In case of non-ensemble training, an ensemble of 1 model behaves like 1 model
                model.eval()
                val_metrics = {
                    name : trainer.validate(model, data_loaders[name], ckpt_path=best_model_path)
                    for name in (data_constants.VAL_TRAIN_LABELS, data_constants.VAL_REDUCED)
                }
                for name, metrics in val_metrics.items():
                    for val_metric in metrics:
                        for metric, value in val_metric.items():
                            result[f'{metric}-{name}-{ensemble_idx}'].append(value)


                # Build evaluation pipeline
                pipeline = Pipeline(config['evaluation']['pipeline'], config['evaluation'], gpus=gpus, 
                    ignore_exceptions=config['evaluation']['ignore_exceptions'])

                # Run evaluation pipeline
                print(f'Executing pipeline...')
                if config['evaluation']['print_pipeline']:
                    print(str(pipeline))
                logs = defaultdict(dict)
                
                # Create a group that will just log the split and initaliation idx
                logs[SPLIT_INIT_GROUP] = {
                    NAME_SPLIT : split_idx, NAME_INIT : reinitialization 
                }

                pipeline_metrics = {} # Metrics logged by the pipeline
                pipeline(
                    model=model.eval(),
                    data_loaders = data_loaders,
                    logger=logger,
                    config=config,
                    artifact_directory=artifact_dir,
                    split_idx = split_idx,
                    initialization_idx = reinitialization,
                    logs=logs,
                    metrics=pipeline_metrics,
                    ensembles=ensembles,
                    artifacts=all_artifacts,
                )

                all_logs.append(logs)
                for metric, value in pipeline_metrics.items():
                    result[metric].append(value)
            
            plt.close('all')

        # Build wandb table for everything that was logged by the pipeline
        build_table(logger, all_logs)
        metrics_path = osp.join(run_artifact_dir, 'metrics.json')
        with open(metrics_path, 'w+') as f:
            json.dump({metric : values for metric, values in result.items()}, f)
        finish_logging(logger)

        # Remove artifacts if they were not to be logged
        if not config['evaluation']['save_artifacts']:
            print(f'Deleting pipeline artifacts...')
            for path in all_artifacts:
                os.remove(path)

        # To not bloat the MongoDB logs, we just return a path to all metrics
        return metrics_path


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
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

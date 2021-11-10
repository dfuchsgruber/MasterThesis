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

from util import suppress_stdout, format_name
from model.util import module_numel
from model.gnn import make_model_by_configuration
from model.semi_supervised_node_classification import SemiSupervisedNodeClassification
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

    def __init__(self, init_all=True, collection_name=None, run_id=None):
        if init_all:
            self.init_all()
        self.collection_name = collection_name
        self.run_id = run_id

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @ex.capture(prefix="data")
    def init_dataset(self, dataset, num_dataset_splits, train_portion, val_portion, test_portion, test_portion_fixed,
                        train_labels='all', train_labels_remove_other=False, train_labels_compress=True,
                        val_labels='all', val_labels_remove_other=False, val_labels_compress=True,
                        split_type='stratified',
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
            'train_labels_compress' : train_labels_compress,
            'val_labels' : val_labels,
            'val_labels_remove_other' : val_labels_remove_other,
            'val_labels_compress' : val_labels_compress,
            'split_type' : split_type,
        }
        # self.data_mask_split, self.data_mask_test_fixed = stratified_split_with_fixed_test_set_portion(self.data[0].y.numpy(), num_dataset_splits, 
        #     portion_train=train_portion, portion_val=val_portion, portion_test_fixed=test_portion_fixed, portion_test_not_fixed=test_portion)

    @ex.capture(prefix="model")
    def init_model(self, model_type: str, hidden_sizes: list, weight_scale: float, num_initializations: int, use_spectral_norm: bool, num_heads=-1, 
        diffusion_iterations=5, teleportation_probability=0.1, use_bias=True, activation='leaky_relu', leaky_relu_slope=0.01, normalize=True,
        residual=False, freeze_residual_projection=False):
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
        }
        self.model_seeds = model_seeds(num_initializations, model_name=model_type)

    def init_all(self):
        """ Sequentially run the sub-initializers of the experiment. """
        self.init_dataset()
        self.init_model()
        self.init_evaluation()
        self.init_run()

    @ex.capture(prefix='evaluation')
    def init_evaluation(self, pipeline=[], print_pipeline=False):
        self.evaluation_config = {
            'pipeline' : pipeline,
            'print_pipeline' : print_pipeline,
        }

    @ex.capture(prefix='run')
    def init_run(self, name='', args=[]):
        self.run_name_format = name
        self.run_name_format_args = args

    @ex.capture(prefix="training")
    def train(self, max_epochs, learning_rate, early_stopping, gpus):
                
        # Data loading
        data_list, dataset_fixed = load_data_from_configuration(self.data_config)

        # Setup config and name of the run(s)
        config = {
            'model' : self.model_config,
            'data' : self.data_config,
            'evaluation' : self.evaluation_config,
            'training' : {
                'max_epochs' : max_epochs,
                'learning_rate' : learning_rate,
                'early_stopping' : early_stopping,
                'gpus' : gpus,
                }
            }
        run_name = format_name(self.run_name_format, self.run_name_format_args, config)
        # One global logger for all splits and initializations
        logger = WandbLogger(save_dir=osp.join('/nfs/students/fuchsgru/wandb'), project=str(self.collection_name), name=f'{run_name}')
        logger.log_hyperparams(config)
        all_logs = [] # Logs from each run

        # Iterating over all dataset splits
        result = defaultdict(list)
        for split_idx, data_dict in enumerate(data_list):

            data_loaders = {
                name : DataLoader(data, batch_size=1, shuffle=False) for name, data in data_dict.items()
            }

            # Re-initializing the model multiple times to average over results
            for reinitialization, seed in enumerate(self.model_seeds):
                pl.seed_everything(seed)

                backbone = make_model_by_configuration(self.model_config, data_get_num_attributes(data_dict[data_constants.TRAIN][0]), data_get_num_classes(data_dict[data_constants.TRAIN][0]))
                model = SemiSupervisedNodeClassification(backbone, learning_rate=learning_rate)
                print(f'Model parameters (trainable / all): {module_numel(model, only_trainable=True)} / {module_numel(model, only_trainable=False)}')

                artifact_dir = osp.join('/nfs/students/fuchsgru/artifacts', str(self.collection_name), f'{run_name}', f'{split_idx}-{reinitialization}')
                os.makedirs(artifact_dir, exist_ok=True)

                # Model training
                with suppress_stdout(), warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    trainer = pl.Trainer(max_epochs=max_epochs, deterministic=True, callbacks=[
                                            ModelCheckpoint(
                                                artifact_dir,
                                                monitor=early_stopping['monitor'],
                                                mode=early_stopping['mode'],
                                                save_top_k=1,
                                            ),
                                            EarlyStopping(
                                                monitor=early_stopping['monitor'],
                                                mode=early_stopping['mode'],
                                                patience=early_stopping['patience'],
                                                min_delta=early_stopping['min_delta'],
                                            ),
                                        ],
                                        logger=logger,
                                        progress_bar_refresh_rate=0,
                                        gpus=gpus,
                                        )
                    trainer.fit(model, data_loaders[data_constants.TRAIN], data_loaders[data_constants.VAL_REDUCED])
                    val_metrics = {
                        name : trainer.validate(None, data_loaders[name], ckpt_path='best')
                        for name in (data_constants.VAL_TRAIN_LABELS, data_constants.VAL_REDUCED)
                    }
                    for name, metrics in val_metrics.items():
                        with open(osp.join(artifact_dir, f'metrics-{name}.json'), 'w+') as f:
                            json.dump(val_metrics, f)
                        for val_metric in metrics:
                            for metric, value in val_metric.items():
                                result[f'{metric}-{name}'].append(value)

                # Build evaluation pipeline
                pipeline = Pipeline(self.evaluation_config['pipeline'], self.evaluation_config, gpus=gpus, ignore_exceptions=True)

                # Run evaluation pipeline
                print(f'Executing pipeline...')
                if self.evaluation_config['print_pipeline']:
                    print(str(pipeline))
                logs = defaultdict(dict)
                
                # Create a group that will just log the split and initaliation idx
                logs[SPLIT_INIT_GROUP] = {
                    NAME_SPLIT : split_idx, NAME_INIT : reinitialization 
                } 
                pipeline_metrics = {} # Metrics logged by the pipeline
                pipeline(
                    model=model,
                    data_loaders = data_loaders,
                    logger=logger,
                    config=config,
                    artifact_directory=artifact_dir,
                    split_idx = split_idx,
                    initialization_idx = reinitialization,
                    logs=logs,
                    metrics=pipeline_metrics, 
                )
                all_logs.append(logs)
                for metric, value in pipeline_metrics.items():
                    result[metric].append(value)
            
            plt.close('all')

        # Build wandb table for everything that was logged by the pipeline
        build_table(logger, all_logs)

        artifact_dir = osp.join('/nfs/students/fuchsgru/artifacts', str(self.collection_name), str(run_name))
        metrics_path = osp.join(artifact_dir, 'metrics.json')
        with open(metrics_path, 'w+') as f:
            json.dump({metric : values for metric, values in result.items()}, f)

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

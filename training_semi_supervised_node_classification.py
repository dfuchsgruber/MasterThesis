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

from util import suppress_stdout
from model.train import train_model_semi_supervised_node_classification
from model.gnn import make_model_by_configuration
from model.semi_supervised_node_classification import SemiSupervisedNodeClassification
from data.gust_dataset import GustDataset
from data.util import SplitDataset, data_get_num_attributes, data_get_num_classes, stratified_split_with_fixed_test_set_portion
from seed import model_seeds
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
    def init_dataset(self, dataset, num_dataset_splits, train_portion, val_portion, test_portion, test_portion_fixed):
        self.data = GustDataset(dataset)
        self.data_config = {
            'dataset' : dataset,
            'num_dataset_splits' : num_dataset_splits,
            'train_portion' : train_portion,
            'val_portion' : val_portion,
            'test_portion' : test_portion,
            'test_portion_fix' : test_portion_fixed,
        }
        self.data_mask_split, self.data_mask_test_fixed = stratified_split_with_fixed_test_set_portion(self.data[0].y.numpy(), num_dataset_splits, 
            portion_train=train_portion, portion_val=val_portion, portion_test_fixed=test_portion_fixed, portion_test_not_fixed=test_portion)

    @ex.capture(prefix="model")
    def init_model(self, model_type: str, hidden_sizes: list, weight_scale: float, num_initializations: int, use_spectral_norm: bool, num_heads=-1, 
        diffusion_iterations=5, teleportation_probability=0.1, use_bias=True, activation='leaky_relu', leaky_relu_slope=0.01, normalize=True):
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
        }
        self.model_seeds = model_seeds(num_initializations, model_name=model_type)

    def init_all(self):
        """ Sequentially run the sub-initializers of the experiment. """
        self.init_dataset()
        self.init_model()

    @ex.capture(prefix="training")
    def train(self, max_epochs, learning_rate, early_stopping, gpus):

        result = defaultdict(list)
        for split_idx in range(self.data_mask_split.shape[1]):
            mask_train, mask_val = self.data_mask_split[0, split_idx], self.data_mask_split[1, split_idx]
            for reinitialization, seed in enumerate(self.model_seeds):

                pl.seed_everything(seed)
                data_loader_train = DataLoader(SplitDataset(self.data, mask_train), batch_size=1, shuffle=False)
                data_loader_val = DataLoader(SplitDataset(self.data, mask_val), batch_size=1, shuffle=False)
                
                backbone = make_model_by_configuration(self.model_config, data_get_num_attributes(self.data[0]), data_get_num_classes(self.data[0]))
                model = SemiSupervisedNodeClassification(backbone, learning_rate=learning_rate)

                # Setup logging and checkpointing
                
                artifact_dir = osp.join('/nfs/students/fuchsgru/artifacts', str(self.collection_name), str(self.run_id), f'{split_idx}-{reinitialization}')
                logger = TensorBoardLogger(osp.join('/nfs/students/fuchsgru/tensorboard', str(self.collection_name), str(self.run_id)), name=f'{split_idx}-{reinitialization}')
                logger.log_hyperparams({
                    'model' : self.model_config,
                    'data' : self.data_config,
                    'training' : {
                        'max_epochs' : max_epochs,
                        'learning_rate' : learning_rate,
                        'early_stopping' : early_stopping,
                        'gpus' : gpus,
                    },
                    'seed' : seed,
                    'initialization_idx' : reinitialization,
                    'split_idx' : split_idx,
                    }
                )

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
                                        weights_summary=None,
                                        gpus=gpus,
                                        )
                    trainer.fit(model, data_loader_train, data_loader_val)
                    val_metrics = trainer.validate(None, data_loader_val, ckpt_path='best')
                    for val_metric in val_metrics:
                        for metric, value in val_metric.items():
                            result[metric].append(value)

        with open(osp.join(artifact_dir, 'metrics.json'), 'w+') as f:
            json.dump({metric : values for metric, values in result.items()}, f)

        return {f'{metric}_mean' : np.array(values).mean() for metric, values in result.items()}


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

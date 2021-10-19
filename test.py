import warnings

import torch
import numpy as np
import os.path as osp
import os

from model.train import train_model_semi_supervised_node_classification
from model.gnn import make_model_by_configuration
from model.semi_supervised_node_classification import SemiSupervisedNodeClassification
from data.gust_dataset import GustDataset
from data.util import data_get_num_attributes, data_get_num_classes, stratified_split_with_fixed_test_set_portion, SplitDataset
from seed import model_seeds
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from util import suppress_stdout
from pytorch_lightning.loggers import WandbLogger
import contextlib, os
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


# mlflow_logger = MLFlowLogger(experiment_name="week2test", tracking_uri="/nfs/students/fuchsgru/mlflow")
wandb_logger = WandbLogger(project="my-test-project", name='foo')


data = GustDataset('cora_ml')
mask_split, mask_fixed = stratified_split_with_fixed_test_set_portion(data[0].y.numpy(), 5, portion_train=0.05, portion_val=0.15, portion_test_fixed=0.2, portion_test_not_fixed=0.6)
data_loader_train = DataLoader(SplitDataset(data, mask_split[0, 0]) , batch_size=1, shuffle=False)
data_loader_val = DataLoader(SplitDataset(data, mask_split[1, 0]) , batch_size=1, shuffle=False)

d = SplitDataset(data, mask_split[1, 0])

# Build the model
cnfg = {
    'model_type' : 'mlp',
    'hidden_sizes' : [64],
    'use_bias' : True,
    'use_spectral_norm' : True,
    'weight_scale' : 5.0,
    'activation' : 'leaky_relu',
    'num_heads' : 2,
    'normalize' : True,
    'diffusion_iterations' : 2,
    'teleportation_probability' : 0.1,
}

wandb_logger.experiment.config.update(cnfg, allow_val_change=True)

model_checkpoint_callback = ModelCheckpoint(
    osp.join('/nfs/students/fuchsgru/artifacts', wandb_logger.experiment.project, wandb_logger.experiment.id),
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    verbose=True,
)

with warnings.catch_warnings():

    warnings.filterwarnings("ignore")
    pl.seed_everything(1337)

    backbone = make_model_by_configuration(cnfg, data_get_num_attributes(data[0]), data_get_num_classes(data[0]))
    gnn = SemiSupervisedNodeClassification(backbone, learning_rate=1e-3)
    trainer = pl.Trainer(max_epochs=50, deterministic=True, log_every_n_steps=1, 
        callbacks=[
            EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=False, min_delta=1e-3),
            model_checkpoint_callback,
        ],
        logger=wandb_logger,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        )
    trainer.fit(gnn, data_loader_train, data_loader_val)
    val_res = (trainer.validate(None, data_loader_val, ckpt_path='best'))

print(val_res)
# data = GustDataset('cora_ml')[0]
# mask_split, mask_fixed = stratified_split_with_fixed_test_set_portion(data.y.numpy(), 5, portion_train=0.05, portion_val=0.15, portion_test_fixed=0.2, portion_test_not_fixed=0.6)

# model_name = 'gcn'
# model_seeds = model_seeds(50, model_name=model_name)
# torch.manual_seed(model_seeds[0])

# model = GNN(model_name, data_get_num_attributes(data), [64], data_get_num_classes(data), use_spectral_norm=True, activation='leaky_relu', 
#     upper_lipschitz_bound=5, num_heads=8, diffusion_iterations=5)
# if torch.cuda.is_available():
#     model.cuda()

# train_history, val_history, best_model_state_dict, best_epoch = train_model_semi_supervised_node_classification(model, data, mask_split[0, 0], mask_split[1, 0], 
#     epochs=1000, early_stopping_patience=100, 
#     early_stopping_metrics={'val_loss' : 'min'}, learning_rate=1e-3)

# for metric, history in val_history.items():
#     print(f'Val {metric} of best model {history[best_epoch]}')
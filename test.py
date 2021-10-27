import warnings

import torch
import numpy as np
import os.path as osp
import os

from model.train import train_model_semi_supervised_node_classification
from model.gnn import make_model_by_configuration
from model.semi_supervised_node_classification import SemiSupervisedNodeClassification
from data.gust_dataset import GustDataset
from data.util import data_get_num_attributes, data_get_num_classes
from seed import model_seeds
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from util import suppress_stdout
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import contextlib, os
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from training_semi_supervised_node_classification import ExperimentWrapper

# data_list, data_test_fixed = load_data_from_configuration({
#     'dataset' : 'cora_ml',
#     'num_dataset_splits' : 1,
#     'train_portion' : 20,
#     'test_portion' : 0.6,
#     'val_portion' : 25,
#     'test_portion_fixed' : 0.2,
#     'train_labels_remove_other' : True,
#     'val_labels_remove_other' : False,
#     'split_type' : 'uniform',
#     'train_labels' : 'all',
#     'val_labels' : 'all',
# })

ex = ExperimentWrapper(init_all=False, collection_name='model_test', run_id='1')
ex.init_dataset(dataset='cora_ml', num_dataset_splits=1, train_portion=20, val_portion=20, test_portion=0.6, test_portion_fixed=0.2,
                    train_labels=[0, 1], val_labels='all', train_labels_remove_other=True, val_labels_remove_other=False,
                    split_type='uniform',
                    )
ex.init_model(model_type='gcn', hidden_sizes=[64], num_initializations=1, weight_scale=5.0, use_spectral_norm=True, use_bias=True, activation='leaky_relu', leaky_relu_slope=0.01)
ex.init_evaluation(pipeline=['FitLogitSpacePCA', 'EvaluateEmpircalLowerLipschitzBounds', 'FitLogitDensityGMM', 'EvaluateLogitDensity', 'LogLogits'], 
    perturbations = {
        'num' : 2,
        'min' : 0.1,
        'max' : 5.0,
        'num_per_sample' : 1,
        'seed' : 1337,
        },
    )

ex.train(max_epochs=500, learning_rate=0.010, early_stopping={
    'monitor' : 'val_loss',
    'mode' : 'min',
    'patience' : 50,
    'min_delta' : 1e-3,
}, gpus=1)

import warnings

import torch
import numpy as np
import os.path as osp
import os
import json

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

ex = ExperimentWrapper(init_all=False, collection_name='model_test', run_id='1')
ex.init_dataset(dataset='cora_ml', num_dataset_splits=1, train_portion=20, val_portion=20, test_portion=0.6, test_portion_fixed=0.2,
                    train_labels=[0, 1, 6, 3, 4, 5], val_labels='all', train_labels_remove_other=True, val_labels_remove_other=False,
                    split_type='uniform',
                    )
# ex.init_dataset(dataset='cora_ml', num_dataset_splits=1, train_portion=0.05, val_portion=0.15, test_portion=0.6, test_portion_fixed=0.2,
#                     train_labels=[0, 1, 2, 3, 4, 5], val_labels='all', train_labels_remove_other=False, val_labels_remove_other=False,
#                     split_type='stratified',
#                     )
ex.init_model(model_type='gcn', hidden_sizes=[64,4], num_initializations=1, weight_scale=5.0, 
    use_spectral_norm=True, use_bias=True, activation='leaky_relu', leaky_relu_slope=0.01,
    residual=False, freeze_residual_projection=False)
ex.init_run(name='model_{0}_hidden_sizes_{1}_weight_scale_{2}', args=[
    'model.model_type', 'model.hidden_sizes', 'model.weight_scale'
])
ex.init_evaluation(pipeline=[
    # {
    #     'type' : 'EvaluateEmpircalLowerLipschitzBounds',
    #     'num_perturbations' : 5,
    #     'min_perturbation' : 2,
    #     'max_perturbation' : 10,
    #     'num_perturbations_per_sample' : 1,
    #     'perturbation_type' : 'noise',
    #     'seed' : 1337,
    #     'name' : 'noise_perturbations',
    # },
    # {
    #     'type' : 'EvaluateEmpircalLowerLipschitzBounds',
    #     'num_perturbations' : 10,
    #     'num_perturbations_per_sample' : 1,
    #     'permute_per_sample' : True,
    #     'perturbation_type' : 'derangement',
    #     'seed' : 1337,
    #     'name' : 'derangement_perturbations',
    # },
    {
        'type' : 'PrintDatasetSummary',
        'evaluate_on' : ['train', 'val', 'val-reduced', 'test', 'test-reduced']
    },
    {
        'type' : 'FitFeatureSpacePCA',
        'fit_to' : ['train'],
        'evaluate_on' : ['train', 'val'],
        'num_components' : 2,
        'name' : '2d-pca',
    },
    {
        'type' : 'FitFeatureDensity',
        'density_type' : 'GaussianPerClass',
        'pca' : False,
        'pca_number_components' : 2,
        'pca_per_class' : False,
        'diagonal_covariance' : True,
        'fit_to' : ['train', 'val-reduced', 'test-reduced'],
        'fit_to_ground_truth_labels' : ['train']
    },
    # {
    #     'type' : 'FitFeatureDensity',
    #     'density_type' : 'GaussianMixture',
    #     'fit_to' : ['train'],
    #     'number_components' : 7,
    # },
    {
        'type' : 'EvaluateFeatureDensity',
        'evaluate_on' : ['val'],
        'name' : 'gaussianperclass'
    },
    {
        'type' : 'LogFeatures',
        'evaluate_on' : ['train', 'val']
    },
    # {
    #     'type' : 'FitFeatureSpacePCAIDvsOOD',
    #     'fit_to' : ['train'],
    #     'evaluate_on' : ['val'],
    # }
]
)

results_path = (ex.train(max_epochs=5, learning_rate=0.001, early_stopping={
    'monitor' : 'val_loss',
    'mode' : 'min',
    'patience' : 50,
    'min_delta' : 1e-3,
}, gpus=1))

with open(results_path) as f:
    print(json.load(f))
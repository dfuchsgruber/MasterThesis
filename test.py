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

from data.construct import uniform_split_with_fixed_test_set_portion
from data.util import label_binarize, data_get_summary

ex = ExperimentWrapper(init_all=False, collection_name='inductive-shift', run_id='gcn_64_32_residual')
ex.init_dataset(dataset='cora_ml', num_dataset_splits=1, train_portion=20, val_portion=20, test_portion=0.6, test_portion_fixed=0.2,
                    train_labels=[0, 1, 6, 3, 4, 5], val_labels='all', train_labels_remove_other=True, val_labels_remove_other=False,
                    split_type='uniform',
                    )
# ex.init_dataset(dataset='cora_ml', num_dataset_splits=1, train_portion=0.05, val_portion=0.15, test_portion=0.6, test_portion_fixed=0.2,
#                     train_labels=[0, 1, 2, 3, 4, 5], val_labels='all', train_labels_remove_other=False, val_labels_remove_other=False,
#                     split_type='stratified',
#                     )
ex.init_model(model_type='gcn', hidden_sizes=[64,32], num_initializations=1, weight_scale=1.0, 
    use_spectral_norm=True, use_bias=True, activation='leaky_relu', leaky_relu_slope=0.01,
    residual=True, freeze_residual_projection=False)
ex.init_run(name='model_no_remove_{0}_hidden_sizes_{1}_weight_scale_{2}', args=[
    'model:model_type', 'model:hidden_sizes', 'model:weight_scale'
])
ex.init_evaluation(
    print_pipeline=True,
    pipeline=[
    # {
    #     'type' : 'EvaluateEmpircalLowerLipschitzBounds',
    #     'num_perturbations' : 20,
    #     'min_perturbation' : 2,
    #     'max_perturbation' : 10,
    #     'num_perturbations_per_sample' : 5,
    #     'perturbation_type' : 'noise',
    #     'seed' : 1337,
    #     'name' : 'noise_perturbations',
    # },
    # {
    #     'type' : 'EvaluateEmpircalLowerLipschitzBounds',
    #     'num_perturbations' : 20,
    #     'num_perturbations_per_sample' : 5,
    #     'permute_per_sample' : True,
    #     'perturbation_type' : 'derangement',
    #     'seed' : 1337,
    #     'name' : 'derangement_perturbations',
    # },
    # {
    #     'type' : 'FitFeatureSpacePCA',
    #     'fit_to' : ['train', 'val-reduced'],
    #     'evaluate_on' : ['train', 'val', 'val-reduced'],
    #     'num_components' : 2,
    #     'name' : '2d-pca',
    # },
    # {
    #     'type' : 'FitFeatureDensity',
    #     'density_type' : 'GaussianMixture',
    #     'dimensionality_reduction' : {
    #         'per_class' : False
    #     },
    #     'fit_to' : [],
    #     'fit_to_ground_truth_labels' : ['train'],
    #     'evaluate_on' : ['val'],
    #     'pipeline_grid' : {
    #         'dimensionality_reduction:type' : ['pca', 'isomap'],
    #         'dimensionality_reduction:number_components' : [3, 4, 8, 16],
    #         'fit_to' : [['train'], ['train', 'val-reduced', 'test-reduced']],
    #         'number_components' : [2, 3, ]
    #     },
    #     'name' : 'mog{3}-{0}-{1}-{2}',
    #     'name_args' : [
    #         'dimensionality_reduction:type',
    #         'dimensionality_reduction:number_components',
    #         'fit_to',
    #         'number_components',
    #     ]
    # },
    # {
    #     'type' : 'FitFeatureDensity',
    #     'density_type' : 'GaussianPerClass',
    #     'dimensionality_reduction' : {
    #         'type' : 'Undefined',
    #     },
    #     'fit_to' : [],
    #     'fit_to_ground_truth_labels' : ['train'],
    #     'evaluate_on' : ['val'],
    #     'pipeline_grid' : {
    #         'diagonal_covariance' : [True,],
    #         'dimensionality_reduction:type' : ['pca', 'isomap', 'none'],
    #         'dimensionality_reduction:per_class': [True, False],
    #         'dimensionality_reduction:number_components': [2, 4, 8, 16, 32],
    #         'fit_to' : [['train'], ['train', 'val-reduced', 'test-reduced']],
    #     },
    #     'name' : 'gpc-{0}-{3}{1}-{2}-pc{4}',
    #     'name_args' : [
    #         'dimensionality_reduction:type',
    #         'diagonal_covariance',
    #         'fit_to',
    #         'dimensionality_reduction:number_components',
    #         'dimensionality_reduction:per_class',
    #     ]
    # },
    {
        'type' : 'PrintDatasetSummary',
        'evaluate_on' : ['train', 'val', 'val-reduced', 'val-train-labels', 'test']
    },
    {
        'type' : 'FitFeatureDensity',
        'density_type' : 'GaussianPerClass',
        'dimensionality_reduction' : {
            'type' : 'pca',
            'per_class' : False,
            'number_components' : 8,
        },
        'fit_to' : ['train'],
        'fit_to_ground_truth_labels' : ['train'],
        'evaluate_on' : ['val'],
        'diagonal_covariance' : True,
        'name' : 'gpc-{0}-{3}{1}-{2}-pc{4}',
        'name_args' : [
            'dimensionality_reduction:type',
            'diagonal_covariance',
            'fit_to',
            'dimensionality_reduction:number_components',
            'dimensionality_reduction:per_class',
        ]
    },
    {
        'type' : 'LogInductiveFeatureShift',
        'data_before' : 'train',
        'data_after' : 'val',
    },
]
)

results_path = (ex.train(max_epochs=1000, learning_rate=0.001, early_stopping={
    'monitor' : 'val_loss',
    'mode' : 'min',
    'patience' : 50,
    'min_delta' : 1e-3,
}, gpus=1))

with open(results_path) as f:
    print(json.load(f))
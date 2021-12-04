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

# ex.init_dataset(dataset='cora_ml', num_dataset_splits=1, train_portion=0.05, val_portion=0.15, test_portion=0.6, test_portion_fixed=0.2,
#                     train_labels=[0, 1, 2, 3, 4, 5], val_labels='all', train_labels_remove_other=False, val_labels_remove_other=False,
#                     split_type='stratified',
#                     )

num_splits, num_inits = 1, 1


ex = ExperimentWrapper(init_all=False, collection_name='model-test', run_id='gcn_64_32_residual')
# ex.init_dataset(dataset='cora_ml', num_dataset_splits=num_splits, train_portion=20, val_portion=20, test_portion=0.6, test_portion_fixed=0.2,
#                     train_labels_remove_other=True, val_labels_remove_other=True,
#                     split_type='uniform',
#                     train_labels = [
#                         'Artificial_Intelligence/Machine_Learning/Case-Based', 
#                         'Artificial_Intelligence/Machine_Learning/Theory', 
#                         # 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 
#                         'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 
#                         'Artificial_Intelligence/Machine_Learning/Neural_Networks',
#                         'Artificial_Intelligence/Machine_Learning/Rule_Learning',
#                         # 'Artificial_Intelligence/Machine_Learning/Reinforcement_Learning',
#                     ], 
#                     # val_labels = 'all',
#                     val_labels = [
#                         'Artificial_Intelligence/Machine_Learning/Case-Based', 
#                         'Artificial_Intelligence/Machine_Learning/Theory', 
#                         # 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 
#                         #'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 
#                         #'Artificial_Intelligence/Machine_Learning/Neural_Networks',
#                         #'Artificial_Intelligence/Machine_Learning/Rule_Learning',
#                         'Artificial_Intelligence/Machine_Learning/Reinforcement_Learning',
#                     ], 
#                     base_labels = 'all',
# )
ex.init_dataset(dataset='cora_full', num_dataset_splits=num_splits, train_portion=20, val_portion=20, test_portion=0.6, test_portion_fixed=0.2,
                    train_labels_remove_other=False, val_labels_remove_other=False,
                    split_type='uniform',
                    base_labels = ['Artificial_Intelligence/NLP', 'Artificial_Intelligence/Data_Mining','Artificial_Intelligence/Speech', 'Artificial_Intelligence/Knowledge_Representation','Artificial_Intelligence/Theorem_Proving', 'Artificial_Intelligence/Games_and_Search','Artificial_Intelligence/Vision_and_Pattern_Recognition', 'Artificial_Intelligence/Planning','Artificial_Intelligence/Agents','Artificial_Intelligence/Robotics', 'Artificial_Intelligence/Expert_Systems','Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning','Operating_Systems/Distributed', 'Operating_Systems/Memory_Management', 'Operating_Systems/Realtime', 'Operating_Systems/Fault_Tolerance'],
                    train_labels = ['Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning'],
                    val_labels = ['Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning','Operating_Systems/Distributed', 'Operating_Systems/Memory_Management', 'Operating_Systems/Realtime', 'Operating_Systems/Fault_Tolerance'],
                    )

ex.init_model(model_type='gcn', hidden_sizes=[64,32], num_initializations=num_inits, weight_scale=2.0, 
    use_spectral_norm=True, use_bias=True, activation='leaky_relu', leaky_relu_slope=0.01,
    residual=True, freeze_residual_projection=False, num_ensemble_members=1, num_samples=1,
    use_spectral_norm_on_last_layer=False,
    #dropout=0.5, drop_edge=0.5,
    )
ex.init_run(name='model_no_remove_{0}_hidden_sizes_{1}_weight_scale_{2}', args=[
    'model:model_type', 'model:hidden_sizes', 'model:weight_scale',
])
ex.init_evaluation(
    print_pipeline=True,
    pipeline=[
    # {
    #     'type' : 'FitFeatureSpacePCAIDvsOOD',
    #     'fit_to' : ['train'],
    #     'separate_distributions_by' : 'neighbourhood',
    #     'separate_distributions_tolerance' : 0.1,
    #     'kind' : 'leave_out_classes',
    # },
    # {
    #     'type' : 'EvaluateEmpircalLowerLipschitzBounds',
    #     'num_perturbations' : 2,
    #     'num_perturbations_per_sample' : 2,
    #     'permute_per_sample' : True,
    #     'perturbation_type' : 'derangement',
    #     'seed' : 1337,
    #     'name' : 'derangement_perturbations',
    # },
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
    #     'type' : 'FitFeatureSpacePCA',
    #     'fit_to' : ['train', 'val-reduced'],
    #     'evaluate_on' : ['train', 'val', 'val-reduced'],
    #     'num_components' : 2,
    #     'name' : '2d-pca',
    # },
    {
        'type' : 'PerturbData',
        'base_data' : 'val-reduced',
        'dataset_name' : 'val-reduced-bernoulli',
        'perturbation_type' : 'bernoulli',
        'budget' : 0.1,
        'parameters' : {
            'p' : 0.5,
        },
        'perturb_in_mask_only' : True,
    },
    {
        'type' : 'PerturbData',
        'base_data' : 'val-reduced',
        'dataset_name' : 'val-reduced-normal',
        'perturbation_type' : 'normal',
        'budget' : 0.1,
        'parameters' : {
            'scale' : 1.0,
        },
        'perturb_in_mask_only' : True,
    },
    {
        'type' : 'EvaluateAccuracy',
        'evaluate_on' : ['val-reduced'],
        'model_kwargs' : {'remove_edges' : True},
        'name' : 'no-edges',
    },
    {
        'type' : 'EvaluateAccuracy',
        'evaluate_on' : ['val-reduced-normal'],
        'kind' : 'perturbations',
    },
    {
        'type' : 'EvaluateAccuracy',
        'evaluate_on' : ['val-reduced-normal'],
        'kind' : 'perturbations',
        'model_kwargs' : {'remove_edges' : True},
        'name' : 'no-edges',
    },
    {
        'type' : 'EvaluateCalibration',
        'evaluate_on' : ['val-reduced'],

    },
    # {
    #     'type' : 'EvaluateAccuracy',
    #     'evaluate_on' : ['val-reduced'],
    #     'model_kwargs' : {'remove_edges' : True},
    #     'name' : 'no-edges',
    # },
    # {
    #     'type' : 'EvaluateAccuracy',
    #     'evaluate_on' : ['val-reduced-ber'],
    # },
    # {
    #     'type' : 'EvaluateAccuracy',
    #     'evaluate_on' : ['val'],
    #     'separate_distributions_by' : 'neighbourhood',
    #     'separate_distributions_tolerance' : 0.1,
    #     'kind' : 'leave_out_classes',
    #     'name' : 'loc',
    # },
    # {
    #     'type' : 'PrintDatasetSummary',
    #     'evaluate_on' : ['train', 'val-reduced', 'val-reduced-ber'],
    # },
    # {
    #     'type' : 'EvaluateSoftmaxEntropy',
    #     'evaluate_on' : ['val-reduced-ber'],
    #     'kind' : 'perturbations',
    # },
    # {
    #     'type' : 'PrintDatasetSummary',
    #     'evaluate_on' : ['train', 'val', 'val-reduced', 'val-train-labels', 'test', 'test-reduced', 'test-train-labels']
    # },
    {
        'type' : 'EvaluateSoftmaxEntropy',
        'evaluate_on' : ['val'],
        'separate_distributions_by' : 'neighbourhood',
        'separate_distributions_tolerance' : 0.1,
        'kind' : 'leave_out_classes',
        'name' : 'loc'
    },
    {
        'type' : 'EvaluateSoftmaxEntropy',
        'name' : 'loc-no-edges',
        'evaluate_on' : ['val'],
        'separate_distributions_by' : 'neighbourhood',
        'separate_distributions_tolerance' : 0.1,
        'kind' : 'leave_out_classes',
        'model_kwargs_evaluate' : {'remove_edges' : True},
    },
    {
        'type' : 'EvaluateSoftmaxEntropy',
        'evaluate_on' : ['val-reduced-normal'],
        'kind' : 'perturbations',
        'name' : 'normal'
    },
    {
        'type' : 'EvaluateSoftmaxEntropy',
        'name' : 'normal-no-edges',
        'evaluate_on' : ['val-reduced-normal'],
        'kind' : 'perturbations',
        'model_kwargs_evaluate' : {'remove_edges' : True},
    },
    # {
    #     'type' : 'EvaluateLogitEnergy',
    #     'evaluate_on' : ['val'],
    #     'separate_distributions_by' : 'neighbourhood',
    #     'separate_distributions_tolerance' : 0.1,
    #     'kind' : 'leave_out_classes',
    # },
    # {
    #     'type' : 'EvaluateLogitEnergy',
    #     'name' : 'no-edges',
    #     'evaluate_on' : ['val'],
    #     'separate_distributions_by' : 'neighbourhood',
    #     'separate_distributions_tolerance' : 0.1,
    #     'kind' : 'leave_out_classes',
    #     'model_kwargs_evaluate' : {'remove_edges' : True},
    # },
    # {
    #     'type' : 'LogInductiveFeatureShift',
    #     'data_before' : 'train',
    #     'data_after' : 'val',
    # },
    # {
    #     'type' : 'LogInductiveSoftmaxEntropyShift',
    #     'data_before' : 'train',
    #     'data_after' : 'val',
    # },
    {
        'type' : 'FitFeatureDensityGrid',
        'fit_to' : ['train'],
        'fit_to_ground_truth_labels' : ['train'],
        'evaluate_on' : ['val'],
        'density_types' : {
            'GaussianPerClass' : {
                'diagonal_covariance' : [True],
            },
        },
        'dimensionality_reductions' : {
            'isomap' : {
                'number_components' : [24],
            },
            'none' : {
            }
        },
        'separate_distributions_by' : 'neighbourhood',
        'separate_distributions_tolerance' : 0.1,
        'kind' : 'leave_out_classes',
        'log_plots' : True,
        'name' : 'loc',
    },
    {
        'type' : 'FitFeatureDensityGrid',
        'fit_to' : ['train'],
        'fit_to_ground_truth_labels' : ['train'],
        'evaluate_on' : ['val'],
        'density_types' : {
            'GaussianPerClass' : {
                'diagonal_covariance' : [True],
            },
        },
        'dimensionality_reductions' : {
            'isomap' : {
                'number_components' : [24],
            },
            'none' : {
            }
        },
        'separate_distributions_by' : 'neighbourhood',
        'separate_distributions_tolerance' : 0.1,
        'kind' : 'leave_out_classes',
        'log_plots' : True,
        'model_kwargs_evaluate' : {'remove_edges' : True},
        'name' : 'loc-no-edges',
    },
    {
        'type' : 'FitFeatureDensityGrid',
        'fit_to' : ['train'],
        'fit_to_ground_truth_labels' : ['train'],
        'evaluate_on' : ['val-reduced-normal'],
        'density_types' : {
            'GaussianPerClass' : {
                'diagonal_covariance' : [True],
            },
        },
        'dimensionality_reductions' : {
            'isomap' : {
                'number_components' : [24],
            },
            'none' : {
            }
        },
        'kind' : 'perturbations',
        'log_plots' : True,
        'name' : 'normal',
    },
    {
        'type' : 'FitFeatureDensityGrid',
        'fit_to' : ['train'],
        'fit_to_ground_truth_labels' : ['train'],
        'evaluate_on' : ['val-reduced-normal'],
        'density_types' : {
            'GaussianPerClass' : {
                'diagonal_covariance' : [True],
            },
        },
        'dimensionality_reductions' : {
            'isomap' : {
                'number_components' : [24],
            },
            'none' : {
            }
        },
        'kind' : 'perturbations',
        'log_plots' : True,
        'model_kwargs_evaluate' : {'remove_edges' : True},
        'name' : 'normal-no-edges',
    },
    # {
    #     'type' : 'FitFeatureDensityGrid',
    #     'name' : 'no-edges',
    #     'fit_to' : ['train'],
    #     'fit_to_ground_truth_labels' : ['train'],
    #     'evaluate_on' : ['val'],
    #     'density_types' : {
    #         'GaussianPerClass' : {
    #             'diagonal_covariance' : [True],
    #         },
    #     },
    #     'dimensionality_reductions' : {
    #         'isomap' : {
    #             'number_components' : [24],
    #         },
    #         'none' : {
    #         }
    #     },
    #     'separate_distributions_by' : 'neighbourhood',
    #     'separate_distributions_tolerance' : 0.1,
    #     'kind' : 'leave_out_classes',
    #     'log_plots' : True,
    #     'model_kwargs_evaluate' : {'remove_edges' : True},
    # },
    ],
    ignore_exceptions=False,
)

results_path = (ex.train(max_epochs=1000, learning_rate=0.001, early_stopping={
    'monitor' : 'val_loss',
    'mode' : 'min',
    'patience' : 50,
    'min_delta' : 1e-3,
}, gpus=0, suppress_stdout=False))

with open(results_path) as f:
    print(json.load(f))
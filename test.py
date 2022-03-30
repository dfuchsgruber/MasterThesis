
import json
from training_semi_supervised_node_classification import ExperimentWrapper
import data.constants as dconstants
import model.constants as mconstants
import configuration
import pandas as pd
import numpy as np
import logging

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f'Test debug message.')

    split_idx, init_idx = 0, 0

    ex = ExperimentWrapper(init_all=False, collection_name='test', run_id='gcn_64_32_residual')
    data_cfg = configuration.DataConfiguration(
                        dataset=dconstants.CORA_FULL, 
                        train_portion=20, test_portion_fixed=0.2,
                        split_type='uniform',
                        type='npz',
                        preprocessing='none',
                        ood_type = dconstants.LEFT_OUT_CLASSES,
                        # ood_type = dconstants.PERTURBATION,
                        # setting = dconstants.HYBRID,
                        setting = dconstants.HYBRID,
                        #preprocessing='word_embedding',
                        #language_model = 'bert-base-uncased',
                        #language_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        #language_model = 'allenai/longformer-base-4096',
                        drop_train_vertices_portion = 0.1,
                        # feature_scale=20.0,
                        )

    if data_cfg.ood_type == dconstants.PERTURBATION:
        data_cfg.left_out_class_labels = []
        data_cfg.base_labels = data_cfg.train_labels
        data_cfg.corpus_labels = data_cfg.train_labels

    spectral_norm_cfg = {
        'use_spectral_norm' : True,
        'residual' : True,
        'weight_scale' : 10.0,
        'use_spectral_norm_on_last_layer' : False,
        # 'self_loop_fill_value' : 5.0,
    }

    reconstruction_cfg = {
        'loss_weight' : 1.0, # Use for BCE-based  and mconstants.ENERGY
        #'loss_weight' : 10.0, # Use higher values for distance based mconstants.TRIPLET
        'sample' : True,
        'num_samples' : 1000,
        'reconstruction_type' : mconstants.ENERGY,
        'margin_constrastive_loss' : 0.0,
    }

    feature_reconstruction_cfg = {
        'loss_weight' : 1.0,
        'loss' : 'weighted_bce',
        'log_metrics_every' : 5,
        'num_samples' : 100,
    }

    bjorck_cfg = {
        # 'weight_scale' : 1.0,
        'orthogonal' : {
            'random_input_projection' : False,
            'bjorck_orthonormalzation_n_iter' : 3,
            'bjorck_orthonormalzation_rescaling' : 10.0,
        },
        'use_bjorck_norm' : True,
        'residual' : True,
    }

    orthonormal_reg = {
        'orthonormal_weight_regularization_strength' : 1e-0,
        'orthonormal_weight_scale' : 10.0,
    }

    bounded_svd = {
        'singular_value_bounding' : True,
        'singular_value_bounding_eps' : 1.0,
    }

    forbenius_norm_cfg = {
        'use_forbenius_norm_on_last_layer' : True,
        'forbenius_norm' : 20.0,
        'residual' : True,
    }

    init_scale_cfg = {
        'initialization_scale' : 50.0,
        'residual' : True,
    }

    rescaling_cfg = {
        'residual' : True,
        'use_rescaling_on_last_layer' : False,
        'weight_scale' : 40.0,
        'use_rescaling' : True,

    }

    spectral_last_cfg = {
        'use_spectral_norm_on_last_layer' : True,
        'weight_scale' : 10,
        # 'use_spectral_norm' : True,
        'residual' : True,
    }

    model_cfg = configuration.ModelConfiguration(
        model_type=mconstants.GCN,
        # model_type = mconstants.BGCN,
        #model_type = 'appnp',
        # dropout = 0.5,
        # drop_edge = 0.2,
        hidden_sizes=[64,], 
        use_bias=True, 
        activation='leaky_relu', 
        leaky_relu_slope=0.01,
        freeze_residual_projection=False, 
        # use_spectral_norm_on_last_layer=True, 
        # self_loop_fill_value=1.0, 
        cached=True,
        # bgcn = {},
        # reconstruction = reconstruction_cfg,
        # feature_reconstruction = feature_reconstruction_cfg,
        # **spectral_norm_cfg,
        #residual = True,
        # **orthogonal_cfg,
        # **forbenius_norm_cfg,
        # **bjorck_cfg,
        #  **init_scale_cfg,
         #**rescaling_cfg,
        # **spectral_last_cfg,
        )
    
    # if not model_cfg.use_spectral_norm:
    #     model_cfg.use_spectral_norm_on_last_layer = False

    run_cfg = configuration.RunConfiguration(
        name='{0}-{1}-sn:{2}-son:{3}-{4}', 
        args=[
            'data:dataset', 'model:model_type', 'model:use_spectral_norm', 'model:use_spectral_norm_on_last_layer', 'model:residual', 'data:setting',
        ],
        split_idx = split_idx,
        initialization_idx = init_idx,
        use_pretrained_model = False,
        use_default_configuration = True,
    )

    self_training = {
        'self_training' : True,
        'num_warmup_epochs' : 100,
    }

    training_cfg = configuration.TrainingConfiguration(
        max_epochs=1000, 
        # min_epochs=500,
        learning_rate=1e-3, 
        early_stopping={
            'monitor' : 'val_loss',
            'mode' : 'min',
            'patience' : 50,
            'min_delta' : 1e-2,
        }, 
        gpus=1, 
        suppress_stdout=False, 
        weight_decay=1e-3,
        # **orthonormal_reg,
        # **self_training,
        #**bounded_svd,
        )

    ensemble_cfg = configuration.EnsembleConfiguration(
        num_members = 1,
        num_samples = 1,
    )

    logging_cfg = configuration.LoggingConfiguration(
        # log_gradients = {
        #     # 'backbone.convs.0.conv.lin.parametrizations.weight.original' : 'conv0_weight',
        #     # 'backbone.convs.1.conv.lin.parametrizations.weight.original' : 'conv1_weight',
        #     # 'backbone.convs.0.input_projection.linear.parametrizations.weight.original' : 'residual0_weight'

        #     'backbone.convs.0.conv.lin.parametrizations.weight.original' : 'conv0_weight',
        #     'backbone.convs.1.conv.lin.parametrizations.weight.original' : 'conv1_weight',
        #     'backbone.convs.0.input_projection.linear.parametrizations.weight.original' : 'residual0_weight'
        # },
        # log_gradients_relative_to_parameter = True,
        # log_gradients_relative_to_norm = True,
        log_weight_matrix_spectrum_every = 1,
    )

    if data_cfg.ood_type == dconstants.PERTURBATION:
        perturbation_pipeline = [
            {
                'type' : 'PerturbData',
                'base_data' : 'ood-val',
                'dataset_name' : 'ood-val-ber',
                'perturbation_type' : 'bernoulli',
                'budget' : 0.1,
                'parameters' : {
                    'p' : 0.5,
                },
                'perturb_in_mask_only' : True,
            },
            {
                'type' : 'PerturbData',
                'base_data' : 'ood-val',
                'dataset_name' : 'ood-val-normal',
                'perturbation_type' : 'normal',
                'budget' : 0.1,
                'parameters' : {
                    'scale' : 1.0,
                },
                'perturb_in_mask_only' : True,
            },
        ]
        ood_separation = 'ood'
        ood_datasets = {'ber' : 'ood-val-ber', }#'normal' : 'ood-val-normal'}
    elif data_cfg.ood_type == dconstants.LEFT_OUT_CLASSES:
        perturbation_pipeline = []
        ood_separation = 'ood-and-neighbourhood'
        ood_datasets = {'loc' : 'ood-val'}
    else:
        raise ValueError(data_cfg.ood_type)


    ood_pipeline = []
    for ood_name, ood_dataset in ood_datasets.items():
        if True: # data_cfg.ood_type != dconstants.LEFT_OUT_CLASSES:
            ood_pipeline += [{
                'type' : 'ValidateAndTest',
                'evaluate_on' : [ood_dataset],
            },]
        ood_pipeline += [
            {
                'type' : 'EvaluateAccuracy',
                'evaluate_on' : ['val'],
            },
            {
                'type' : 'FitFeatureDensityGrid',
                'fit_to' : ['train'],
                'fit_to_ground_truth_labels' : ['train'],
                'fit_to_mask_only' : True,
                'fit_to_best_prediction' : False,
                'fit_to_min_confidence' : 0.99,
                'evaluate_on' : [ood_dataset],
                'diffuse_features' : False,
                'diffusion_iterations' : 16,
                'teleportation_probability' : 0.2,
                'density_types' : {
                    'GaussianPerClass' : {
                        'evaluation_kwargs_grid' : [{'mode' : ['weighted'], 'relative' : [False,]}],
                        'covariance' : ['full', 'diag', 'eye', 'iso'],
                    },
                    # 'GaussianMixture' : {
                    #     'diagonal_covariance' : [True],
                    #     'number_components' : [-1],
                    # }
                },
                'dimensionality_reductions' : {
                    'none' : {
                    }
                },
                'log_plots' : True,
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'name' : ood_name,
            },
            {
                'type' : 'FitFeatureDensityGrid',
                'fit_to' : ['train'],
                'fit_to_ground_truth_labels' : ['train'],
                'fit_to_mask_only' : True,
                'fit_to_best_prediction' : False,
                'fit_to_min_confidence' : 0.99,
                'evaluate_on' : [ood_dataset],
                'diffuse_features' : False,
                'diffusion_iterations' : 16,
                'teleportation_probability' : 0.2,
                'density_types' : {
                    'GaussianPerClass' : {
                        'evaluation_kwargs_grid' : [{'mode' : ['weighted'], 'relative' : [False,]}],
                        'covariance' : ['full', 'diag', 'eye', 'iso'],
                    },
                    # 'GaussianMixture' : {
                    #     'diagonal_covariance' : [True],
                    #     'number_components' : [-1],
                    # }
                },
                'dimensionality_reductions' : {
                    'none' : {
                    }
                },
                'log_plots' : True,
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'name' : f'{ood_name}-no-edges',
                'model_kwargs_evaluate' : {'remove_edges' : True},
            },
            # {
            #     'type' : 'FitFeatureDensityGrid',
            #     'fit_to' : ['train'],
            #     'fit_to_ground_truth_labels' : ['train', 'val'],
            #     'fit_to_mask_only' : True,
            #     'fit_to_best_prediction' : False,
            #     'fit_to_min_confidence' : 0.95,
            #     'evaluate_on' : [ood_dataset],
            #     'diffuse_features' : False,
            #     'diffusion_iterations' : 16,
            #     'teleportation_probability' : 0.2,
            #     'density_types' : {
            #         'NormalizingFlowPerClass' : {
            #             'evaluation_kwargs_grid' : [{'mode' : ['weighted'], 'relative' : [False,]}],
            #             'flow_type' : ['radial'],
            #             'num_layers' : [5, 10,]
            #         },
            #     },
            #     'dimensionality_reductions' : {
            #         'none' : {
            #         }
            #     },
            #     'log_plots' : True,
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #     'name' : ood_name,
            # },
            # {
            #     'type' : 'FitFeatureDensityGrid',
            #     'fit_to' : ['train'],
            #     'fit_to_ground_truth_labels' : ['train', 'val'],
            #     'fit_to_mask_only' : True,
            #     'fit_to_best_prediction' : False,
            #     'fit_to_min_confidence' : 0.95,
            #     'evaluate_on' : [ood_dataset],
            #     'diffuse_features' : False,
            #     'diffusion_iterations' : 16,
            #     'teleportation_probability' : 0.2,
            #     'density_types' : {
            #         'NormalizingFlowPerClass' : {
            #             'evaluation_kwargs_grid' : [{'mode' : ['weighted'], 'relative' : [False,]}],
            #             'flow_type' : ['maf'],
            #             'num_layers' : [2, 5, 10, 15]
            #         },
            #     },
            #     'dimensionality_reductions' : {
            #         'none' : {
            #         }
            #     },
            #     'log_plots' : True,
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #     'name' : ood_name,
            # },
            # {
            #     'type' : 'EvaluateFeatureSpaceDistance',
            #     'fit_to' : ['train'],
            #     'evaluate_on' : [ood_dataset],
            #     'log_plots' : True,
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #     'k' : 5,
            #     'layer' : -2,
            #     'name' : ood_name,
            # },
            # {
            #     'type' : 'EvaluateFeatureSpaceDistance',
            #     'fit_to' : ['train'],
            #     'evaluate_on' : [ood_dataset],
            #     'log_plots' : True,
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #     'k' : 5,
            #     'layer' : 0,
            #     'name' : 'input-' + ood_name
            # },
            # {
            #     'type' : 'EvaluateStructure',
            #     'fit_to' : ['train'],
            #     'evaluate_on' : [ood_dataset],
            #     'log_plots' : True,
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #     'diffusion_iterations' : 16,
            #     'teleportation_probability' : 0.2,
            #     'name' : ood_name,
            # },
            # {
            #     'type' : 'EvaluateAccuracy',
            #     'evaluate_on' : [ood_dataset],
            #     'model_kwargs' : {'remove_edges' : False},
            #     'name' : ood_name,
            # },
            # {
            #     'type' : 'EvaluateLogitGeometry',
            #     'evaluate_on' : [ood_dataset],
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #      'name' : ood_name,
            # },
            {
                'type' : 'EvaluateSoftmaxEntropy',
                'evaluate_on' : [ood_dataset],
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'name' : ood_name,
            },
            {
                'type' : 'EvaluateSoftmaxEntropy',
                'evaluate_on' : [ood_dataset],
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'name' : f'{ood_name}-no-edges',
                'model_kwargs_evaluate' : {'remove_edges' : True},

            },
            {
                'type' : 'EvaluateFeatureSpaceDistance',
                'fit_to' : ['train'],
                'evaluate_on' : [ood_dataset],
                'log_plots' : True,
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'k' : 5,
                'layer' : -2,
                'name' : ood_name,
            },
            {
                'type' : 'EvaluateFeatureSpaceDistance',
                'fit_to' : ['train'],
                'evaluate_on' : [ood_dataset],
                'log_plots' : True,
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'k' : 5,
                'layer' : -2,
                'name' : f'{ood_name}-no-edges',
                'model_kwargs_evaluate' : {'remove_edges' : True},
            },
            # {
            #     'type' : 'EvaluateLogitEnergy',
            #     'evaluate_on' : [ood_dataset],
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #     'kind' : 'leave_out_classes',
            #     'name' : ood_name,
            # },
            # {
            #     'type' : 'VisualizeIDvsOOD',
            #     'fit_to' : ['train'],
            #     'evalaute_on' : [ood_dataset],
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #     'kind' : 'leave_out_classes',
            #     'dimensionality_reductions' : ['pca', 'tsne',],
            #     'name' : ood_name,
            # },
        ]

    evaluation_cfg = configuration.EvaluationConfiguration(
        print_pipeline = True,
        ignore_exceptions = False,
        log_plots = True,
        pipeline = perturbation_pipeline + ood_pipeline + [
            {
                'type' : 'EvaluateAccuracy',
                'evaluate_on' : ['val'],
            },
            {
                'type' : 'EvaluateCalibration',
                'evaluate_on' : ['val'],
            },
        ],
        # sample = True,
    )


    out = ex.train(model=model_cfg, data = data_cfg, training=training_cfg, run=run_cfg, evaluation=evaluation_cfg, ensemble=ensemble_cfg, logging=logging_cfg)

    print()
    df = pd.DataFrame({k : {'Mean' : np.mean(v), 'Std' : np.std(v)} for k, v in out['results'].items()})
    print(df.T.to_markdown())
    print()

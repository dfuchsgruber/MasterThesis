
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

    split_idx, init_idx = 1, 0

    ex = ExperimentWrapper(init_all=False, collection_name='test', run_id='gcn_64_32_residual')
    data_cfg = configuration.DataConfiguration(
                        dataset='cora_full', 
                        train_portion=20, test_portion_fixed=0.2,
                        split_type='uniform',
                        type='npz',
                        base_labels = ['Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning','Operating_Systems/Distributed', 'Operating_Systems/Memory_Management', 'Operating_Systems/Realtime', 'Operating_Systems/Fault_Tolerance'],
                        train_labels = ['Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning'],
                        left_out_class_labels = ['Operating_Systems/Distributed', 'Operating_Systems/Memory_Management', 'Operating_Systems/Realtime', 'Operating_Systems/Fault_Tolerance'],
                        corpus_labels = ['Artificial_Intelligence/Machine_Learning/Case-Based', 'Artificial_Intelligence/Machine_Learning/Theory', 'Artificial_Intelligence/Machine_Learning/Genetic_Algorithms', 'Artificial_Intelligence/Machine_Learning/Probabilistic_Methods', 'Artificial_Intelligence/Machine_Learning/Neural_Networks','Artificial_Intelligence/Machine_Learning/Rule_Learning','Artificial_Intelligence/Machine_Learning/Reinforcement_Learning'],
                        preprocessing='bag_of_words',
                        ood_type = dconstants.LEFT_OUT_CLASSES,
                        # ood_type = dconstants.PERTURBATION,
                        setting = dconstants.HYBRID,
                        #preprocessing='word_embedding',
                        #language_model = 'bert-base-uncased',
                        #language_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        #language_model = 'allenai/longformer-base-4096',
                        drop_train_vertices_portion = 0.1,
                        ood_sampling_strategy = dconstants.SAMPLE_ALL,
                        )

    spectral_norm_cfg = {
        'use_spectral_norm' : True,
        'residual' : True,
        'weight_scale' : 0.5,
    }

    reconstruction_cfg = {
        'loss_weight' : 1.0, # Use for BCE-based  and mconstants.ENERGY)
        #'loss_weight' : 10.0, # Use higher values for distance based (mconstants.TRIPLET
        'sample' : True,
        'num_samples' : 1000,
        'reconstruction_type' : mconstants.ENERGY,
        'margin_constrastive_loss' : 0.0,
    }



    model_cfg = configuration.ModelConfiguration(
        model_type=mconstants.GCN,
        #model_type = 'appnp',
        diffusion_iterations = 16,
        teleportation_probability = 0.2,
        hidden_sizes=[64], 
        use_bias=True, 
        activation='leaky_relu', 
        leaky_relu_slope=0.01,
        freeze_residual_projection=False, 
        use_spectral_norm_on_last_layer=False, 
        self_loop_fill_value=1.0, 
        cached=True,
        reconstruction = reconstruction_cfg,
        # **spectral_norm_cfg,
        )

    run_cfg = configuration.RunConfiguration(
        name='model_{0}_hidden_sizes_{1}_weight_scale_{2}_setting_{3}_ood_type_{4}', 
        args=[
            'model:model_type', 'model:hidden_sizes', 'model:weight_scale', 'data:setting', 'data:ood_type',
        ],
        model_registry_collection_name = 'model_registry',
        split_idx = split_idx,
        initialization_idx = init_idx,
        # use_pretrained_model = False,
    )

    self_training = {
        'self_training' : True,
        'num_warmup_epochs' : 100,
    }

    training_cfg = configuration.TrainingConfiguration(
        max_epochs=1000, # 1000, 
        learning_rate=0.001, 
        early_stopping={
            'monitor' : 'val_loss',
            'mode' : 'min',
            'patience' : 50,
            'min_delta' : 1e-2,
        }, 
        gpus=1, 
        suppress_stdout=False, 
        weight_decay=1e-3,
        # **self_training,
        )

    ensemble_cfg = configuration.EnsembleConfiguration(
        num_members = 1,
    )

    ood_separation = 'ood' # 'ood-and-neighbourhood'

    evaluation_cfg = configuration.EvaluationConfiguration(
        print_pipeline = True,
        ignore_exceptions = False,
        log_plots = True,
        pipeline = [
            {
                'type' : 'FitFeatureDensityGrid',
                'fit_to' : ['train'],
                'fit_to_ground_truth_labels' : ['train'],
                'fit_to_mask_only' : True,
                'fit_to_best_prediction' : False,
                'fit_to_min_confidence' : 0.99,
                'evaluate_on' : ['ood-val'],
                'diffuse_features' : False,
                'diffusion_iterations' : 16,
                'teleportation_probability' : 0.2,
                'density_types' : {
                    'GaussianPerClass' : {
                        'evaluation_kwargs_grid' : [{'mode' : ['max', 'weighted'], 'relative' : [False, True]}],
                        'covariance' : ['full', 'diag', 'eye', 'iso'],
                    },
                    'GaussianMixture' : {
                        'diagonal_covariance' : [True],
                        'number_components' : [-1],
                    }
                },
                'dimensionality_reductions' : {
                    'none' : {
                    }
                },
                'log_plots' : True,
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
            },
            {
                'type' : 'EvaluateFeatureSpaceDistance',
                'fit_to' : ['train'],
                'evaluate_on' : ['ood-val'],
                'log_plots' : True,
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'k' : 5,
                'layer' : -2,
            },
            # {
            #     'type' : 'EvaluateFeatureSpaceDistance',
            #     'fit_to' : ['train'],
            #     'evaluate_on' : ['ood-val'],
            #     'log_plots' : True,
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #     'k' : 5,
            #     'layer' : 0,
            #     'name' : 'input'
            # },
            # {
            #     'type' : 'EvaluateStructure',
            #     'fit_to' : ['train'],
            #     'evaluate_on' : ['ood-val'],
            #     'log_plots' : True,
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #     'diffusion_iterations' : 16,
            #     'teleportation_probability' : 0.2,
            # },
            # {
            #     'type' : 'EvaluateAccuracy',
            #     'evaluate_on' : [dconstants.OOD_VAL],
            #     'model_kwargs' : {'remove_edges' : False},
            # },
            # {
            #     'type' : 'EvaluateLogitGeometry',
            #     'evaluate_on' : ['ood-val'],
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            # },
            # {
            #     'type' : 'PrintDatasetSummary',
            #     'evaluate_on' : ['train', 'val', 'ood-val'],
            # },
            {
                'type' : 'EvaluateSoftmaxEntropy',
                'evaluate_on' : ['ood-val'],
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
            },
            # {
            #     'type' : 'EvaluateLogitEnergy',
            #     'evaluate_on' : ['ood-val'],
            #     'separate_distributions_by' : ood_separation,
            #     'separate_distributions_tolerance' : 0.1,
            #     'kind' : 'leave_out_classes',
            # },
            {
                'type' : 'VisualizeIDvsOOD',
                'fit_to' : ['train'],
                'evalaute_on' : ['ood-val'],
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'kind' : 'leave_out_classes',
                'dimensionality_reductions' : ['pca', 'tsne',],
            },
            # {
            #     'type' : 'LogInductiveFeatureShift',
            #     'data_before' : 'train',
            #     'data_after' : 'ood-val',
            # },
            # {
            #     'type' : 'LogInductiveSoftmaxEntropyShift',
            #     'data_before' : 'train',
            #     'data_after' : 'ood-val',
            # },
            # {
            #     'type' : 'EvaluateCalibration',
            #     'evaluate_on' : ['val'],
            # },
        ],
    )


    out = ex.train(model=model_cfg, data = data_cfg, training=training_cfg, run=run_cfg, evaluation=evaluation_cfg, ensemble=ensemble_cfg)

    print()
    df = pd.DataFrame({k : {'Mean' : np.mean(v), 'Std' : np.std(v)} for k, v in out['results'].items()})
    print(df.T.to_markdown())
    print()

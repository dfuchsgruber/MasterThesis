
import json
from training_semi_supervised_node_classification import ExperimentWrapper
import data.constants as dconstants
import configuration

num_splits, num_inits = 2, 2


ex = ExperimentWrapper(init_all=False, collection_name='model-test', run_id='gcn_64_32_residual')
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

model_cfg = configuration.ModelConfiguration(
    model_type='gcn', 
    hidden_sizes=[64], 
    weight_scale=1.4, 
    use_spectral_norm=False, 
    use_bias=True, 
    activation='leaky_relu', 
    leaky_relu_slope=0.01,
    residual=False, 
    freeze_residual_projection=False, 
    use_spectral_norm_on_last_layer=False, 
    self_loop_fill_value=1.0,
    )

run_cfg = configuration.RunConfiguration(
    name='model_{0}_hidden_sizes_{1}_weight_scale_{2}_setting_{3}_ood_type_{4}', 
    args=[
        'model:model_type', 'model:hidden_sizes', 'model:weight_scale', 'data:setting', 'data:ood_type',
    ],
    model_registry_collection_name = 'test_registry',
    num_initializations = num_inits,
    num_dataset_splits = num_splits,
)

training_cfg = configuration.TrainingConfiguration(
    max_epochs=1000, 
    learning_rate=0.001, 
    early_stopping={
        'monitor' : 'val_loss',
        'mode' : 'min',
        'patience' : 100,
        'min_delta' : 1e-3,
    }, 
    gpus=1, 
    suppress_stdout=False, 
    weight_decay=1e-3)

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
            'type' : 'PrintDatasetSummary',
            'evaluate_on' : ['train', 'val', 'ood-val'],
        },
        {
            'type' : 'VisualizeIDvsOOD',
            'fit_to' : ['train'],
            'evalaute_on' : ['ood-val'],
            'separate_distributions_by' : ood_separation,
            'separate_distributions_tolerance' : 0.1,
            'kind' : 'leave_out_classes',
            'dimensionality_reductions' : ['pca', 'tsne',],
        },
        {
            'type' : 'LogInductiveFeatureShift',
            'data_before' : 'train',
            'data_after' : 'ood-val',
        },
        {
            'type' : 'LogInductiveSoftmaxEntropyShift',
            'data_before' : 'train',
            'data_after' : 'ood-val',
        },
        {
            'type' : 'EvaluateCalibration',
            'evaluate_on' : ['val'],
        },
        {
            'type' : 'EvaluateAccuracy',
            'evaluate_on' : [dconstants.OOD_VAL],
            'model_kwargs' : {'remove_edges' : False},
        },{
            'type' : 'EvaluateSoftmaxEntropy',
            'evaluate_on' : ['ood-val'],
            'separate_distributions_by' : ood_separation,
            'separate_distributions_tolerance' : 0.1,
        },{
            'type' : 'EvaluateLogitEnergy',
            'evaluate_on' : ['ood-val'],
            'separate_distributions_by' : ood_separation,
            'separate_distributions_tolerance' : 0.1,
            'kind' : 'leave_out_classes',
        },{
            'type' : 'FitFeatureDensityGrid',
            'fit_to' : ['train'],
            'fit_to_ground_truth_labels' : ['train'],
            'fit_to_mask_only' : True,
            'fit_to_best_prediction' : False,
            'fit_to_min_confidence' : 0.99,
            'evaluate_on' : ['ood-val'],
            'density_types' : {
                'GaussianPerClass' : {
                    'diagonal_covariance' : [True],
                    'evaluation_kwargs_grid' : [{'mode' : ['weighted',], 'relative' : [False,]}]
                },
            },
            'dimensionality_reductions' : {
                'none' : {
                }
            },
            'log_plots' : True,
            'separate_distributions_by' : ood_separation,
            'separate_distributions_tolerance' : 0.1,
        },
    ],
)


out = ex.train(model=model_cfg, data = data_cfg, training=training_cfg, run=run_cfg, evaluation=evaluation_cfg, ensemble=ensemble_cfg)

with open(out['results']) as f:
    print(json.load(f))
print(out['configuration'])





# evaluation_cfg = configuration.EvaluationConfiguration(
#     save_artifacts=False,
#     print_pipeline=True,
#     pipeline= 
#     [
#     {
#         'type' : 'FitFeatureDensityGrid',
#         'fit_to' : ['train'],
#         'fit_to_ground_truth_labels' : ['train'],
#         'fit_to_mask_only' : False,
#         'fit_to_best_prediction' : False,
#         'fit_to_min_confidence' : 0.99,
#         'evaluate_on' : ['ood-val'],
#         'density_types' : {
#             'GaussianPerClass' : {
#                 'diagonal_covariance' : [True, False],
#             },
#             'NormalizingFlowPerClass' : {
#                 'flow_type' : ['maf'],
#                 'gpu' : [True],
#                 'verbose' : [True],
#                 'weight_decay' : [1e-3],
#             },
#             'NormalizingFlow' : {
#                 'flow_type' : ['maf'],
#                 'gpu' : [True],
#                 'verbose' : [True],
#                 'weight_decay' : [1e-3],
#             },
#             'GaussianMixture' : {
#                 'number_components' : [-1],
#                 'diagonal_covariance' : [True, False],
#                 'initialization' : ['random', 'predictions'],
#             },
#         },
#         'dimensionality_reductions' : {
#             'none' : {
#             }
#         },
#         'log_plots' : True,
#         'separate_distributions_by' : 'ood-and-neighbourhood',
#         'separate_distributions_tolerance' : 0.1,
#     },
#     {
#         'type' : 'EvaluateSoftmaxEntropy',
#         'evaluate_on' : ['ood-val'],
#         'separate_distributions_by' : 'ood-and-neighbourhood',
#         'separate_distributions_tolerance' : 0.1,
#     },
#     # {
#     #     'type' : 'PerturbData',
#     #     'base_data' : 'ood',
#     #     'dataset_name' : 'ood-ber',
#     #     'perturbation_type' : 'bernoulli',
#     #     'parameters' : {
#     #         'p' : 0.5,
#     #     },
#     # },
#     # {
#     #     'type' : 'PerturbData',
#     #     'base_data' : 'ood',
#     #     'dataset_name' : 'ood-normal',
#     #     'perturbation_type' : 'normal',
#     #     'parameters' : {
#     #         'scale' : 1.0,
#     #     },
#     # },
#     # {
#     #     'type' : 'VisualizeIDvsOOD',
#     #     'fit_to' : ['train'],
#     #     'evalaute_on' : ['ood'],
#     #     'separate_distributions_by' : 'ood-and-neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     #     'dimensionality_reductions' : ['pca', 'tsne',],
#     # },
#     # {
#     #     'type' : 'VisualizeIDvsOOD',
#     #     'fit_to' : ['train'],
#     #     'evalaute_on' : ['ood'],
#     #     'separate_distributions_by' : 'ood-and-neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     #     'layer' : -1,
#     #     'name' : 'logits',
#     #     'dimensionality_reductions' : ['pca', 'tsne',],
#     # },
#     # {
#     #     'type' : 'EvaluateEmpircalLowerLipschitzBounds',
#     #     'num_perturbations' : 2,
#     #     'num_perturbations_per_sample' : 2,
#     #     'permute_per_sample' : True,
#     #     'perturbation_type' : 'derangement',
#     #     'seed' : 1337,
#     #     'name' : 'derangement_perturbations',
#     # },
#     # {
#     #     'type' : 'EvaluateEmpircalLowerLipschitzBounds',
#     #     'num_perturbations' : 20,
#     #     'min_perturbation' : 2,
#     #     'max_perturbation' : 10,
#     #     'num_perturbations_per_sample' : 5,
#     #     'perturbation_type' : 'noise',
#     #     'seed' : 1337,
#     #     'name' : 'noise_perturbations',
#     # },
#     # {
#     #     'type' : 'FitFeatureSpacePCA',
#     #     'fit_to' : ['train', 'val'],
#     #     'evaluate_on' : ['train', 'val', 'ood-normal'],
#     #     'num_components' : 2,
#     #     'name' : '2d-pca-normal',
#     # },
#     # {
#     #     'type' : 'FitFeatureSpacePCA',
#     #     'fit_to' : ['train', 'val'],
#     #     'evaluate_on' : ['train', 'val', 'ood-ber'],
#     #     'num_components' : 2,
#     #     'name' : '2d-pca-ber',
#     # },
#     # {
#     #     'type' : 'EvaluateAccuracy',
#     #     'evaluate_on' : ['ood-normal'],
#     #     'separate_distributions_by' : 'ood-and-neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     # },
#     # {
#     #     'type' : 'EvaluateAccuracy',
#     #     'evaluate_on' : ['ood-ber'],
#     #     'separate_distributions_by' : 'ood-and-neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     # },
#     # {
#     #     'type' : 'EvaluateCalibration',
#     #     'evaluate_on' : ['val'],
#     # },
#     # {
#     #     'type' : 'EvaluateSoftmaxEntropy',
#     #     'evaluate_on' : ['ood-normal'],
#     #     'separate_distributions_by' : 'ood-and-neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'name' : 'normal',
#     # },
#     # {
#     #     'type' : 'EvaluateSoftmaxEntropy',
#     #     'evaluate_on' : ['ood-ber'],
#     #     'separate_distributions_by' : 'ood-and-neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'name' : 'ber',
#     # },
#     # {
#     #     'type' : 'FitFeatureDensityGrid',
#     #     'fit_to' : ['train'],
#     #     'fit_to_ground_truth_labels' : ['train'],
#     #     'evaluate_on' : ['ood-normal'],
#     #     'name' : 'normal',
#     #     'density_types' : {
#     #         'GaussianPerClass' : {
#     #             'diagonal_covariance' : [True],
#     #             'relative' : [True, False],
#     #             'mode' : ['weighted', 'max'],
#     #         },
#     #     },
#     #     'dimensionality_reductions' : {
#     #         'pca' : {
#     #             'number_components' : [2],
#     #         },
#     #         'none' : {
#     #         }
#     #     },
#     #     'separate_distributions_by' : 'ood-and-neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     #     'log_plots' : True,
#     # },
#     # {
#     #     'type' : 'FitFeatureDensityGrid',
#     #     'fit_to' : ['train'],
#     #     'fit_to_ground_truth_labels' : ['train'],
#     #     'evaluate_on' : ['ood-ber'],
#     #     'name' : 'ber',
#     #     'density_types' : {
#     #         'GaussianPerClass' : {
#     #             'diagonal_covariance' : [True],
#     #             'relative' : [True, False],
#     #             'mode' : ['weighted', 'max'],
#     #         },
#     #     },
#     #     'dimensionality_reductions' : {
#     #         'pca' : {
#     #             'number_components' : [2],
#     #         },
#     #         'none' : {
#     #         }
#     #     },
#     #     'separate_distributions_by' : 'ood-and-neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     #     'log_plots' : True,
#     # },
#     # {
#     #     'type' : 'EvaluateAccuracy',
#     #     'evaluate_on' : ['val-reduced'],
#     #     'model_kwargs' : {'remove_edges' : True},
#     #     'name' : 'no-edges',
#     # },
#     # {
#     #     'type' : 'EvaluateAccuracy',
#     #     'evaluate_on' : ['val-reduced-ber'],
#     # },
#     # {
#     #     'type' : 'EvaluateAccuracy',
#     #     'evaluate_on' : ['val'],
#     #     'separate_distributions_by' : 'neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     #     'name' : 'loc',
#     # },
#     # {
#     #     'type' : 'PrintDatasetSummary',
#     #     'evaluate_on' : ['train', 'val-reduced', 'val-reduced-ber'],
#     # },
#     # {
#     #     'type' : 'EvaluateSoftmaxEntropy',
#     #     'name' : 'loc-no-edges',
#     #     'evaluate_on' : ['val'],
#     #     'separate_distributions_by' : 'neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     #     'model_kwargs_evaluate' : {'remove_edges' : True},
#     # },
#     # {
#     #     'type' : 'EvaluateSoftmaxEntropy',
#     #     'evaluate_on' : ['val-reduced-normal'],
#     #     'kind' : 'perturbations',
#     #     'name' : 'normal'
#     # },
#     # {
#     #     'type' : 'EvaluateSoftmaxEntropy',
#     #     'name' : 'normal-no-edges',
#     #     'evaluate_on' : ['val-reduced-normal'],
#     #     'kind' : 'perturbations',
#     #     'model_kwargs_evaluate' : {'remove_edges' : True},
#     # },
#     # {
#     #     'type' : 'EvaluateLogitEnergy',
#     #     'evaluate_on' : ['val'],
#     #     'separate_distributions_by' : 'neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     # },
#     # {
#     #     'type' : 'EvaluateLogitEnergy',
#     #     'name' : 'no-edges',
#     #     'evaluate_on' : ['val'],
#     #     'separate_distributions_by' : 'neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     #     'model_kwargs_evaluate' : {'remove_edges' : True},
#     # },
#     # {
#     #     'type' : 'LogInductiveFeatureShift',
#     #     'data_before' : 'train',
#     #     'data_after' : 'val',
#     # },
#     # {
#     #     'type' : 'LogInductiveSoftmaxEntropyShift',
#     #     'data_before' : 'train',
#     #     'data_after' : 'val',
#     # },
#     # {
#     #     'type' : 'FitFeatureDensityGrid',
#     #     'fit_to' : ['train'],
#     #     'fit_to_ground_truth_labels' : ['train'],
#     #     'evaluate_on' : ['val'],
#     #     'density_types' : {
#     #         'GaussianPerClass' : {
#     #             'diagonal_covariance' : [True],
#     #             'relative' : [True, False],
#     #             'mode' : ['weighted', 'max'],
#     #         },
#     #     },
#     #     'dimensionality_reductions' : {
#     #         'none' : {
#     #         }
#     #     },
#     #     'separate_distributions_by' : 'neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     #     'log_plots' : True,
#     #     'name' : 'loc',
#     # },
#     # {
#     #     'type' : 'FitFeatureDensityGrid',
#     #     'fit_to' : ['train'],
#     #     'fit_to_ground_truth_labels' : ['train'],
#     #     'evaluate_on' : ['val'],
#     #     'density_types' : {
#     #         'GaussianPerClass' : {
#     #             'diagonal_covariance' : [True],
#     #             'relative' : [True, False],
#     #             'mode' : ['weighted', 'max'],
#     #         },
#     #     },
#     #     'dimensionality_reductions' : {
#     #         'none' : {
#     #         }
#     #     },
#     #     'separate_distributions_by' : 'neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     #     'log_plots' : True,
#     #     'model_kwargs_evaluate' : {'remove_edges' : True},
#     #     'name' : 'loc-no-edges',
#     # },
#     # {
#     #     'type' : 'FitFeatureDensityGrid',
#     #     'fit_to' : ['train'],
#     #     'fit_to_ground_truth_labels' : ['train'],
#     #     'evaluate_on' : ['val-reduced-normal'],
#     #     'density_types' : {
#     #         'GaussianPerClass' : {
#     #             'diagonal_covariance' : [True],
#     #             'relative' : [True, False],
#     #             'mode' : ['weighted', 'max'],
#     #         },
#     #     },
#     #     'dimensionality_reductions' : {
#     #         'none' : {
#     #         }
#     #     },
#     #     'kind' : 'perturbations',
#     #     'log_plots' : True,
#     #     'name' : 'normal',
#     # },
#     # {
#     #     'type' : 'FitFeatureDensityGrid',
#     #     'fit_to' : ['train'],
#     #     'fit_to_ground_truth_labels' : ['train'],
#     #     'evaluate_on' : ['val-reduced-normal'],
#     #     'density_types' : {
#     #         'GaussianPerClass' : {
#     #             'diagonal_covariance' : [True],
#     #             'relative' : [True, False],
#     #             'mode' : ['weighted', 'max'],
#     #         },
#     #     },
#     #     'dimensionality_reductions' : {
#     #         'none' : {
#     #         }
#     #     },
#     #     'kind' : 'perturbations',
#     #     'log_plots' : True,
#     #     'model_kwargs_evaluate' : {'remove_edges' : True},
#     #     'name' : 'normal-no-edges',
#     # },
#     # {
#     #     'type' : 'FitFeatureDensityGrid',
#     #     'name' : 'no-edges',
#     #     'fit_to' : ['train'],
#     #     'fit_to_ground_truth_labels' : ['train'],
#     #     'evaluate_on' : ['val'],
#     #     'density_types' : {
#     #         'GaussianPerClass' : {
#     #             'diagonal_covariance' : [True],
#     #             'relative' : [True, False],
#     #             'mode' : ['weighted', 'max'],
#     #         },
#     #     },
#     #     'dimensionality_reductions' : {
#     #         'none' : {
#     #         }
#     #     },
#     #     'separate_distributions_by' : 'neighbourhood',
#     #     'separate_distributions_tolerance' : 0.1,
#     #     'kind' : 'leave_out_classes',
#     #     'log_plots' : True,
#     #     'model_kwargs_evaluate' : {'remove_edges' : True},
#     # },
#     ],
#     ignore_exceptions=False,
# )
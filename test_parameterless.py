
import json
from training_semi_supervised_node_classification import ExperimentWrapper
import data.constants as dconstants
import model.constants as mconstants
import configuration
import pandas as pd
import numpy as np
import logging

if __name__ == '__main__':

    split_idx, init_idx = 0, 0

    ex = ExperimentWrapper(init_all=False, collection_name='test', run_id='parameterless')
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
                        # setting = dconstants.TRANSDUCTIVE,
                        #preprocessing='word_embedding',
                        #language_model = 'bert-base-uncased',
                        #language_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        #language_model = 'allenai/longformer-base-4096',
                        drop_train_vertices_portion = 0.1,
                        ood_sampling_strategy = dconstants.SAMPLE_ALL,
                        )

    if data_cfg.ood_type == dconstants.PERTURBATION:
        data_cfg.left_out_class_labels = []
        data_cfg.base_labels = data_cfg.train_labels
        data_cfg.corpus_labels = data_cfg.train_labels




    model_cfg = configuration.ModelConfiguration(
        model_type=mconstants.INPUT_DISTANCE,
        appnp = {
            'diffusion_iterations' : 16,
            'teleportation_probability' : 0.2,
        },
        input_distance = {
            'centroids' : False,
            'k' : 5,
        },
        gdk = {
            'sigma' : 1.0,
            'reduction' : 'max',
        }
        )

    run_cfg = configuration.RunConfiguration(
        name='model_{0}_hidden_sizes_{1}_weight_scale_{2}_setting_{3}_ood_type_{4}', 
        args=[
            'model:model_type', 'model:hidden_sizes', 'model:weight_scale', 'data:setting', 'data:ood_type',
        ],
        split_idx = split_idx,
        initialization_idx = init_idx,
        use_pretrained_model = True,
        use_default_configuration = True,
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
        ood_datasets = {'ber' : 'ood-val-ber', 'normal' : 'ood-val-normal'}
    elif data_cfg.ood_type == dconstants.LEFT_OUT_CLASSES:
        perturbation_pipeline = []
        ood_separation = 'ood-and-neighbourhood'
        ood_datasets = {'loc' : 'ood-val'}
    else:
        raise ValueError(data_cfg.ood_type)


    ood_pipeline = []
    for ood_name, ood_dataset in ood_datasets.items():
        ood_pipeline += [
            {
                'type' : 'UncertaintyQuantificationByPredictionAttribute',
                'evaluate_on' : [ood_dataset],
                'attribute' : 'evidence_total',
                'log_plots' : False,
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'name' : ood_name + '_total',
            },
            {
                'type' : 'UncertaintyQuantificationByPredictionAttribute',
                'evaluate_on' : [ood_dataset],
                'attribute' : 'evidence_prediction',
                'log_plots' : False,
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'name' : ood_name + '_prediction',
            },
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


    out = ex.train(model=model_cfg, data = data_cfg, run=run_cfg, evaluation=evaluation_cfg)

    print()
    df = pd.DataFrame({k : {'Mean' : np.mean(v), 'Std' : np.std(v)} for k, v in out['results'].items()})
    print(df.T.to_markdown())
    print()

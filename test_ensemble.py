
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
                        dataset='cora_full', 
                        train_portion=20, test_portion_fixed=0.2,
                        split_type='uniform',
                        type='npz',
                        preprocessing='none',
                        # ood_type = dconstants.LEFT_OUT_CLASSES,
                        ood_type = dconstants.PERTURBATION,
                        setting = dconstants.HYBRID,
                        # setting = dconstants.TRANSDUCTIVE,
                        #preprocessing='word_embedding',
                        #language_model = 'bert-base-uncased',
                        #language_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        #language_model = 'allenai/longformer-base-4096',
                        drop_train_vertices_portion = 0.1,
                        )

    if data_cfg.ood_type == dconstants.PERTURBATION:
        data_cfg.left_out_class_labels = []
        data_cfg.base_labels = data_cfg.train_labels
        data_cfg.corpus_labels = data_cfg.train_labels

    model_cfg = configuration.ModelConfiguration(
        model_type=mconstants.GCN_LAPLACE,
        # model_type = mconstants.BGCN,
        #model_type = 'appnp',
        # dropout = 0.5,
        # drop_edge = 0.2,
        hidden_sizes=[64,], 
        use_bias=True, 
        activation='leaky_relu', 
        leaky_relu_slope=0.01,
        freeze_residual_projection=False, 
        use_spectral_norm_on_last_layer=False, 
        self_loop_fill_value=1.0, 
        cached=True,
        # bgcn = {},
        # reconstruction = reconstruction_cfg,
        laplace = {
            'hessian_structure' : mconstants.DIAG_HESSIAN,
            'batch_size' : 64,
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
        model_registry_collection_name = 'model_test',
    )


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
        )

    ensemble_cfg = configuration.EnsembleConfiguration(
        num_members = 1,
        num_samples = 10,
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
                'type' : 'EvaluateAccuracy',
                'evaluate_on' : ['val'],
            },
            {
                'type' : 'EvaluateSoftmaxEntropy',
                'evaluate_on' : [ood_dataset],
                'separate_distributions_by' : ood_separation,
                'separate_distributions_tolerance' : 0.1,
                'name' : ood_name,
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
        sample = True,
    )


    out = ex.train(model=model_cfg, data = data_cfg, training=training_cfg, run=run_cfg, evaluation=evaluation_cfg, ensemble=ensemble_cfg)

    print()
    df = pd.DataFrame({k : {'Mean' : np.mean(v), 'Std' : np.std(v)} for k, v in out['results'].items()})
    print(df.T.to_markdown())
    print()

# Get two levels above `__file__` to import thesis modules

from copy import deepcopy
import yaml
import os
import os.path as osp

# Configurations to be copied from the base
BASE_KEYS = ['seml', 'slurm', 'fixed', 'grid']

def build():
    dir = osp.dirname(__file__)
    fn, suffix = osp.splitext(osp.basename(__file__))
    with open(osp.join(dir, fn + '.base.yaml')) as f:
        base = yaml.safe_load(f)

    cfg = {k : base[k] for k in BASE_KEYS}
    build_experiments(cfg)

    outfile = osp.join(dir, fn + '.yaml')
    with open(outfile, 'w+') as f:
        yaml.dump(deepcopy(cfg), f)
    print(f'Wrote configuration file to {outfile}')

NO_EDGES = {
    'model_kwargs' : {'remove_edges' : True},
    'name' : 'no-edges',
}

def build_experiments(cfg):
    for ood_type, ood_type_short in (('left-out-classes', 'loc'), ('perturbations', 'per')):

        # Build an evaluation pipeline for the ood type
        pipeline = [
            {
                'type' : 'EvaluateAccuracy',
                'evaluate_on' : ['val'],
            },
            {
                'type' : 'EvaluateAccuracy',
                'evaluate_on' : ['val'],
            } | deepcopy(NO_EDGES),
            {
                'type' : 'EvaluateCalibration',
                'evaluate_on' : ['val'],
            },
            {
                'type' : 'EvaluateCalibration',
                'evaluate_on' : ['val'],
            } | deepcopy(NO_EDGES),
            {
                'type' : 'EvaluateEmpircalLowerLipschitzBounds',
                'num_perturbations' : 20,
                'min_perturbation' : .1,
                'max_perturbation' : 10.0,
                'num_perturbations_per_sample' : 5,
                'seed' : 1337,
                'perturbation_type' : 'noise',
                'name' : 'noise',
            },
            # {
            #     'type' : 'EvaluateEmpircalLowerLipschitzBounds',
            #     'num_perturbations' : 10,
            #     'min_perturbation' : .1,
            #     'max_perturbation' : 10.0,
            #     'num_perturbations_per_sample' : 5,
            #     'seed' : 1337,
            #     'perturbation_type' : 'derangement',
            #     'name' : 'derangement',
            # },
        ]

        if ood_type == 'perturbations':
            # Build perturbed datasets
            pipeline += [
                {
                    'type' : 'PerturbData',
                    'base_data' : 'ood-val',
                    'dataset_name' : 'ber',
                    'perturbation_type' : 'bernoulli',
                    'parameters' : {
                        'p' : 0.5,
                    },
                },
                {
                    'type' : 'PerturbData',
                    'base_data' : 'ood-val',
                    'dataset_name' : 'normal',
                    'perturbation_type' : 'normal',
                    'parameters' : {
                        'scale' : 1.0,
                    },
                },
            ]
            ood_datasets = {'ber' : 'ber', 'normal' : 'normal'}
        else:
            ood_datasets = {'loc' : 'ood-val'}
        
        for ood_name, ood_dataset in ood_datasets.items():
            if ood_name == 'loc':
                ood_args = {
                    'separate_distributions_by' : 'ood-and-neighbourhood',
                    'separate_distributions_tolerance' : 0.1,
                }
            else:
                ood_args = {}
            for suffix, args in (
                ('-no-edges', {
                'model_kwargs_evaluate' : {'remove_edges' : True}
                }), 
                ('', {})
                ):
                pipeline += [
                    {
                        'type' : 'EvaluateAccuracy',
                        'evaluate_on' : [ood_dataset],
                        'name' : f'{ood_name}{suffix}',
                    } | deepcopy(args) | deepcopy(ood_args),
                    # {
                    #     'type' : 'VisualizeIDvsOOD',
                    #     'fit_to' : ['train'],
                    #     'evaluate_on' : [ood_dataset],
                    #     'dimensionality_reductions': ['pca', 'tsne'],
                    #     'name' : f'{ood_name}{suffix}',
                    # } | deepcopy(args) | deepcopy(ood_args),
                    {
                        'type' : 'EvaluateSoftmaxEntropy',
                        'evaluate_on' : [ood_dataset],
                        'name' : f'{ood_name}{suffix}',
                    } | deepcopy(args) | deepcopy(ood_args),
                    {
                        'type' : 'EvaluateLogitEnergy',
                        'evaluate_on' : [ood_dataset],
                        'name' : f'{ood_name}{suffix}',
                    } | deepcopy(args) | deepcopy(ood_args),
                    # {
                    #     'type' : 'LogInductiveFeatureShift',
                    #     'data_before' : 'train',
                    #     'data_after' : ood_dataset,
                    #     'name' : f'{ood_name}{suffix}',
                    # } | deepcopy(args) | deepcopy(ood_args),
                    # {
                    #     'type' : 'LogInductiveSoftmaxEntropyShift',
                    #     'data_before' : 'train',
                    #     'data_after' : ood_dataset,
                    #     'name' : f'{ood_name}{suffix}',
                    # } | deepcopy(args) | deepcopy(ood_args),
                ]
                pipeline.append(
                    {
                        'type' : 'EvaluateFeatureSpaceDistance',
                        'fit_to' : ['train'],
                        'evaluate_on' : ['ood-val'],
                        'log_plots' : False,
                        'k' : 5,
                        'layer' : -2,
                    } | deepcopy(ood_args) | deepcopy(args)
                )

                for fit_name, fit_args in (
                    ('', {'fit_to_mask_only' : True, 'fit_to_best_prediction' : False, 'fit_to_min_confidence' : 0.0,}),
                    ('-fit-all', {'fit_to_mask_only' : False, 'fit_to_best_prediction' : False, 'fit_to_min_confidence' : 0.0,}),
                    ('-fit-best', {'fit_to_mask_only' : False, 'fit_to_best_prediction' : True,}),
                    ('-fit-95-conf', {'fit_to_mask_only' : False, 'fit_to_best_prediction' : True, 'fit_to_min_confidence' : 0.95}),
                ):
                    pipeline.append({
                        'type' : 'FitFeatureDensityGrid',
                        'evaluate_on' : [ood_dataset],
                        'fit_to' : ['train'],
                        'fit_to_ground_truth_labels': ['train'],
                        'density_types' : {
                            'GaussianPerClass' : {
                                'covariance' : ['full', 'diag', 'eye', 'iso'],
                                'evaluation_kwargs_grid' : [{'mode' : ['weighted', 'max'], 'relative' : [False, True]}],
                            },
                            'GaussianMixture' : {
                            'number_components' : [-1],
                            'diagonal_covariance' : [True, False],
                            'initialization' : ['random', 'predictions'],
                            },
                            'NormalizingFlowPerClass' : {
                                'flow_type' : ['maf', 'radial'],
                                'num_layers' : [2, 5, 10, 20],
                                'iterations' : ['auto'],
                                'evaluation_kwargs_grid' : [{'mode' : ['weighted', 'max'], 'relative' : [False, True]}],
                                'gpu' : [True],
                                'verbose' : [False],
                                'weight_decay' : [1e-3],
                            },
                            'NormalizingFlow' : {
                                'flow_type' : ['maf', 'radial'],
                                'num_layers' : [2, 5, 10, 20],
                                'iterations' : ['auto'],
                                'gpu' : [True],
                                'verbose' : [False],
                                'weight_decay' : [1e-3],
                            }
                        },
                        'dimensionality_reductions' : {
                            'none' : {}
                        },
                        'name' : f'{ood_name}{fit_name}{suffix}',
                        'log_plots' : False,
                    } | deepcopy(args) | deepcopy(ood_args) | deepcopy(fit_args))


        for spectral_norm in (True, False):
            subcfg = {
                'fixed' : {
                    'evaluation.pipeline' : deepcopy(pipeline),
                }
            }
            subcfg['fixed']['model.use_spectral_norm'] = spectral_norm
            if spectral_norm:
                subcfg['fixed'] |= {
                    'model.weight_scale' : 3.0,
                    'model.use_spectral_norm_on_last_layer' : False,
                }
            else:
                subcfg['fixed'] |= {
                    'model.weight_scale' : 1.0,
                    'model.use_spectral_norm_on_last_layer' : False,
                }
                
                
            subcfg['fixed']['run.name'] = 'residual:{0}-spectral_norm-{1}-setting:{2}-ood_type:' + ood_type_short
            subcfg['fixed']['data.ood_type'] = ood_type
            if ood_type == 'perturbations':
                subcfg['fixed']['data.left_out_class_labels'] = []
                subcfg['fixed']['data.base_labels'] = deepcopy(cfg['fixed']['data.train_labels'])
                subcfg['fixed']['data.corpus_labels'] = deepcopy(cfg['fixed']['data.train_labels'])


            cfg[f'{ood_type}' + ('-spectral-norm' if spectral_norm else '-no-spectral-norm')] = subcfg

        
if __name__ == '__main__':
    build()
# Get two levels above `__file__` to import thesis modules

from copy import deepcopy
from re import sub
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

        # Build the evaluation pipeline
        pipeline_all_eval_modes = []

        for eval_mode in ('val', 'test'): # Run evaluation both on validation and test set

            # Build an evaluation pipeline for the ood type
            pipeline = []
            pipeline += [
                {
                    'type' : 'EvaluateAccuracy',
                    'evaluate_on' : [eval_mode],
                },
                {
                    'type' : 'EvaluateAccuracy',
                    'evaluate_on' : [eval_mode],
                } | deepcopy(NO_EDGES),
                {
                    'type' : 'EvaluateCalibration',
                    'evaluate_on' : [eval_mode],
                },
                {
                    'type' : 'EvaluateCalibration',
                    'evaluate_on' : [eval_mode],
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
                    'evaluate_on' : [eval_mode]
                },
                {
                    'type' : 'EvaluateEmpircalLowerLipschitzBounds',
                    'num_perturbations' : 20,
                    'min_perturbation' : .1,
                    'max_perturbation' : 10.0,
                    'num_perturbations_per_sample' : 5,
                    'seed' : 1337,
                    'perturbation_type' : 'noise',
                    'name' : 'noise',
                    'evaluate_on' : [eval_mode]
                } | deepcopy(NO_EDGES),

            ]

            if ood_type == 'perturbations':
                # Build perturbed datasets
                pipeline += [
                    {
                        'type' : 'PerturbData',
                        'base_data' : f'ood-{eval_mode}',
                        'dataset_name' : f'ber-{eval_mode}',
                        'perturbation_type' : 'bernoulli',
                        'parameters' : {
                            'p' : 0.5,
                        },
                    },
                    {
                        'type' : 'PerturbData',
                        'base_data' : f'ood-{eval_mode}',
                        'dataset_name' : f'normal-{eval_mode}',
                        'perturbation_type' : 'normal',
                        'parameters' : {
                            'scale' : 1.0,
                        },
                    },
                ]
                ood_datasets = {'ber' : f'ber-{eval_mode}', 'normal' : f'normal-{eval_mode}'}
            else:
                ood_datasets = {'loc' : f'ood-{eval_mode}'}
            
            for ood_name, ood_dataset in ood_datasets.items():
                if ood_name == 'loc':
                    ood_args = {
                        'separate_distributions_by' : 'ood-and-neighbourhood',
                        'separate_distributions_tolerance' : 0.1,
                    }
                else:
                    ood_args = {}

                pipeline += [
                    {
                        'type' : 'EvaluateFeatureSpaceDistance',
                        'fit_to' : ['train'],
                        'validate_on' : ['val'],
                        'evaluate_on' : [ood_dataset],
                        'log_plots' : False,
                        'k' : 5,
                        'layer' : 0,
                        'name' : 'input-' + ood_name
                    } | ood_args,
                    {
                        'type' : 'EvaluateStructure',
                        'fit_to' : ['train'],
                        'evaluate_on' : [ood_dataset],
                        'log_plots' : False,
                        'diffusion_iterations' : 10,
                        'teleportation_probability' : 0.1,
                        'name' : ood_name,
                    } | ood_args,
                ]

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

                    # Feature space densities
                    # GPC
                    for fitting_suffix, fitting_args in (
                        ('_fit-mask', {'fit_to_mask_only' : True}),
                        ('_fit-95conf', {'fit_to_mask_only' : False, 'fit_to_best_prediction' : False, 'fit_to_min_confidence' : 0.95}),
                    ):
                        pipeline.append({
                            'type' : 'FitFeatureDensityGrid',
                            'fit_to' : ['train'],
                            'fit_to_ground_truth_labels' : ['train', 'val'],
                            'evaluate_on' : [ood_dataset],
                            'diffuse_features' : False,
                            'diffusion_iterations' : 16,
                            'teleportation_probability' : 0.2,
                            'density_types' : {
                                'GaussianPerClass' : {
                                'covariance' : ['full', 'diag', 'eye', 'iso'],
                                'evaluation_kwargs_grid' : [{'mode' : ['weighted', 'max'], 'relative' : [False, True]}],
                                },
                            },
                            'dimensionality_reductions' : {
                                'none' : {
                                }
                            },
                            'log_plots' : False,
                            'name' : f'{ood_name}{fitting_suffix}{suffix}',
                        } | deepcopy(args) | deepcopy(ood_args) | deepcopy(fitting_args))
                        pipeline.append({
                            'type' : 'EvaluateFeatureSpaceDistance',
                            'fit_to' : ['train'],
                            'evaluate_on' : [ood_dataset],
                            'log_plots' : False,
                            'k' : 5,
                            'layer' : -2,
                            'name' : f'{ood_name}{fitting_suffix}{suffix}',
                        } | deepcopy(args) | deepcopy(ood_args))
                        pass

            for member in pipeline:
                member['name'] = member.get('name', '') + f'_{eval_mode}'

            pipeline_all_eval_modes += pipeline


        for spectral_norm, weight_scales in (
            (False, (1.0,)),
            (True, (0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0))):
            for weight_scale in weight_scales:
                subcfg = {
                    'fixed' : {
                        'evaluation.pipeline' : deepcopy(pipeline_all_eval_modes),
                    }
                }
                subcfg['fixed']['model.use_spectral_norm'] = spectral_norm
                subcfg['fixed']['model.weight_scale'] = weight_scale
                    
                subcfg['fixed']['run.name'] = 'residual:{0}-spectral_norm-{1}-weight_scale{3}-setting:{2}-ood_type:' + ood_type_short
                subcfg['fixed']['data.ood_type'] = ood_type
                if ood_type == 'perturbations':
                    subcfg['fixed']['data.left_out_class_labels'] = []
                    subcfg['fixed']['data.base_labels'] = deepcopy(cfg['fixed']['data.train_labels'])
                    subcfg['fixed']['data.corpus_labels'] = deepcopy(cfg['fixed']['data.train_labels'])


                cfg[f'{ood_type}' + ('-spectral-norm' if spectral_norm else '-no-spectral-norm') + f'-weight-scale-{weight_scale}'.replace('.', '-')] = subcfg

        
if __name__ == '__main__':
    build()
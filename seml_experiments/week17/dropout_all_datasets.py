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

    for dataset, jobs in (('cora_full', 16),  ('cora_ml', 16), ('pubmed', 16), ('citeseer', 16), ('coauthor_cs', 8), ('amazon_photo', 8), ('ogbn_arxiv', 4)):
        cfg = deepcopy({k : base[k] for k in BASE_KEYS})
        cfg['fixed']['data.dataset'] = dataset
        cfg['slurm']['experiments_per_job'] = jobs
        if  dataset == 'ogbn_arxiv':
            cfg['grid']['run.split_idx'] = {'type' : 'choice', 'options' : [0]}

        build_experiments(cfg)

        outfile = osp.join(dir, fn, f'{dataset}.yaml')
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        with open(outfile, 'w+') as f:
            yaml.dump(deepcopy(cfg), f)
        print(f'Wrote configuration file to {outfile}')

NO_EDGES = {
    'model_kwargs' : {'remove_edges' : True},
    'name' : 'no-edges',
}


def build_experiments(cfg):
    for ex_prefix, (dropout, drop_edge) in (('dropout', (0.5, 0.0)), ('drop_edge', (0.0, 0.5))):
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
                        ]
                        

                for member in pipeline:
                    member['name'] = member.get('name', '') + f'_{eval_mode}'

                pipeline_all_eval_modes += pipeline

            subcfg = {
                'fixed': {
                    'evaluation.pipeline' : deepcopy(pipeline_all_eval_modes),
                    'run.name' : f'dropout:{0}-drop_edge:{1}-dataset:{3}-setting:{2}-ood_type:' + ood_type_short,
                    'model.dropout' : dropout,
                    'model.drop_edge' : drop_edge,
                    'data.ood_type' : ood_type,
                },
                'grid' : {
                },
            }

            key = f'{ood_type}_{ex_prefix}'
            assert key not in cfg
            cfg[key] = deepcopy(subcfg)
        
if __name__ == '__main__':
    build()
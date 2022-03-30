# This script exports datasets for the hybrid and transductive setting both for LoC and perturbation experiments
# These can then be used by the GPN-fork
# It uses the pipeline to best imitate the procedure that is implemented by this project.
import numpy as np
import json
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os.path as osp
import yaml

from configuration import *
import data.constants as dconstants
from data.build import load_data_from_configuration
from evaluation.pipeline import Pipeline
import seed
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data Export')
    parser.add_argument('-d', dest='directory', default=osp.join('./.exported_datasets'), help='In which directory to export data into.')
    parser.add_argument('datasets', metavar='datasets', type=str, nargs='+', help='Which datasets to export.')
    parser.add_argument('-n', dest='num_splits', default=5)
    parser.add_argument('--no-integrity-assertion', dest='no_integrity_assertion', action='store_true', help='If no integrity assertions should be performed.')
    args = parser.parse_args()
    for dataset in args.datasets:
        if dataset not in dconstants.DATASETS:
            raise RuntimeError(f'Unsupported dataset {dataset}. Supported are {dconstants.DATASETS}')

    for dataset in args.datasets:

        base_config = get_default_configuration_by_dataset(dataset)
        num_splits = 1 if base_config['data']['split_type'] == 'predefined' else args.num_splits

        print(f'Exporting {num_splits} split(s) for dataset {dataset}...')

        for ood_type in (dconstants.LEFT_OUT_CLASSES, dconstants.PERTURBATION):
            for setting in (dconstants.TRANSDUCTIVE, dconstants.HYBRID):

                config = ExperimentConfiguration(data = DataConfiguration(
                                    dataset=dataset, 
                                    ood_type = ood_type,
                                    setting = setting,
                                    integrity_assertion = not args.no_integrity_assertion,
                                    **base_config['data'],
                                ),
                                    model=ModelConfiguration(**base_config['model'])
                                )
                update_with_default_configuration(config)

                # if ood_type == dconstants.PERTURBATION:
                #     config.data.left_out_class_labels = []
                #     config.data.base_labels = config.data.train_labels
                #     config.data.corpus_labels = config.data.train_labels


                for split_idx in tqdm(range(num_splits), desc='Exporting dataset splits...'):

                    data_split_seed = seed.data_split_seeds()[split_idx]

                    data_dict, fixed_vertices = load_data_from_configuration(config.data, data_split_seed)
                    config.evaluation.pipeline = []

                    if config.data.ood_type == dconstants.PERTURBATION:

                        ood_datasets = ['ood-val-normal', 'ood-test-normal', 'ood-val-ber', 'ood-test-ber']
                        # For a perturbation experiment we create datasets for the bernoulli and gaussian perturbation case
                        config.evaluation.pipeline += [
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
                                'base_data' : 'ood-test',
                                'dataset_name' : 'ood-test-ber',
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
                            {
                                'type' : 'PerturbData',
                                'base_data' : 'ood-test',
                                'dataset_name' : 'ood-test-normal',
                                'perturbation_type' : 'normal',
                                'budget' : 0.1,
                                'parameters' : {
                                    'scale' : 1.0,
                                },
                                'perturb_in_mask_only' : True,
                            },
                            ]
                    else:
                        ood_datasets = ['ood-val', 'ood-test']

                    config.evaluation.pipeline += [
                        {
                            'type' : 'ExportData',
                            'datasets' : 'all',
                            'ood_datasets' : ood_datasets,
                            'output_path' : osp.join(args.directory, '{data.dataset}', '{data.setting}-{data.ood_type}', 'split-{run.split_idx}.pkl'),
                            # These settings will only be used when we are in the LoC setting
                            # Note that this depends on the model depth, so you might want to change: config.model.hidden_sizes
                            'separate_distributions_by' : 'ood-and-neighbours',
                            'separate_distributions_tolerance' : 0.1,
                        }
                    ]

                    loaders = {name : DataLoader(dataset, batch_size=1, shuffle=False) for name, dataset in data_dict.items()}
                    config.run.split_idx = split_idx
                    pipeline = Pipeline(config.evaluation.pipeline, config.evaluation, gpus=config.training.gpus, 
                                    ignore_exceptions=config.evaluation.ignore_exceptions)
                    pipeline(data_loaders=loaders, model=None, config=config)

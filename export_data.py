# This script exports datasets for the hybrid and transductive setting both for LoC and perturbation experiments
# These can then be used by the GPN-fork
# It uses the pipeline to best imitate the procedure that is implemented by this project.
import numpy as np
import json
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from configuration import *
import data.constants as dconstants
from data.construct import load_data_from_configuration
from evaluation.pipeline import Pipeline

config = ExperimentConfiguration(data = DataConfiguration(
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
                    ), run = {'num_dataset_splits' : 5}, 
                    model=ModelConfiguration(hidden_sizes=[64,])
                    )

data_list, fixed_vertices = load_data_from_configuration(config.data, config.run.num_dataset_splits)
config.evaluation.pipeline = []

if config.data.ood_type == dconstants.PERTURBATION:

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
            'dataset_name' : 'ood-val-test',
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

config.evaluation.pipeline += [
    {
        'type' : 'ExportData',
        'datasets' : 'all',
        'output_path' : './.exported_datasets/{data.dataset}/{data.setting}-{data.ood_type}/split-{registry.split_idx}.pkl',
        # These settings will only be used when we are in the LoC setting
        # Note that this depends on the model depth, so you might want to change: config.model.hidden_sizes
        'separate_distributions_by' : 'ood-and-neighbours',
        'separate_distributions_tolerance' : 0.1,
    }
]

for split_idx, data_dict in enumerate(tqdm(data_list, desc='Exporting dataset splits...')):
    loaders = {name : DataLoader(dataset, batch_size=1, shuffle=False) for name, dataset in data_dict.items()}
    config.registry.split_idx = split_idx
    pipeline = Pipeline(config.evaluation.pipeline, config.evaluation, gpus=config.training.gpus, 
                    ignore_exceptions=config.evaluation.ignore_exceptions)
    pipeline(data_loaders=loaders, model=None, config=config)

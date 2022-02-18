import torch
import numpy as npf
from torch_geometric.loader import DataLoader
import os, re, pickle
import os.path as osp

from .base import *
from .ood import OODSeparation
import data.constants as dconstants
from evaluation.util import get_data_loader
from data.util import data_get_summary, labels_to_idx, graph_select_labels
from data.base import SingleGraphDataset
from data.transform import PerturbationTransform

@register_pipeline_member
class PrintDatasetSummary(PipelineMember):
    """ Pipeline member that prints dataset statistics. """

    name = 'PrintDatasetSummary'

    def __init__(self, gpus=0, evaluate_on=[dconstants.TRAIN, dconstants.VAL, dconstants.OOD_VAL], **kwargs):
        super().__init__(**kwargs)
        self.evaluate_on = evaluate_on

    @property
    def configuration(self):
        return super().configuration | {
            'Evaluate on' : self.evaluate_on,
        }

    def __call__(self, *args, **kwargs):
        for name in self.evaluate_on:
            loader = get_data_loader(name, kwargs['data_loaders'])
            print(f'# Data summary : {name}')
            print(data_get_summary(loader.dataset, prefix='\t'))

        return args, kwargs

@register_pipeline_member
class SubsetDataByLabel(PipelineMember):
    """ Creates a new data loader that holds a subset of some dataset with only certain labels. """

    name = 'SubsetDataByLabel'

    def __init__(self, base_data=dconstants.OOD_VAL, subset_name='unnamed-subset', labels='all', **kwargs):
        super().__init__(**kwargs)
        self.base_data = base_data
        self.subset_name = subset_name.lower()
        self.labels = labels

    @property
    def configuration(self):
        return super().configuration | {
            'Based on' : self.base_data,
            'Subset name' : self.subset_name,
            'Labels' : self.labels,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        base_loader = get_data_loader(self.base_data, kwargs['data_loaders'])
        if len(base_loader) != 1:
            raise RuntimeError(f'Subsetting by label is only supported for single graph data.')
        data = base_loader.dataset[0]
        labels = labels_to_idx(self.labels, data)
        x, edge_index, y, vertex_to_idx, label_to_idx, mask = graph_select_labels(data.x.numpy(), 
            data.edge_index.numpy(), data.y.numpy(), data.vertex_to_idx, data.label_to_idx, labels, 
            connected=True, _compress_labels=True)
        data = SingleGraphDataset(x, edge_index, y, vertex_to_idx, label_to_idx, np.ones(y.shape[0]).astype(bool))
        kwargs['data_loaders'][self.subset_name] = DataLoader(data, batch_size=1, shuffle=False)
        return args, kwargs

@register_pipeline_member
class PerturbData(PipelineMember):
    """ Pipeline member that creates a dataset with perturbed features. """

    name = 'PerturbData'

    def __init__(self, base_data=dconstants.OOD_VAL, dataset_name='unnamed-perturbation-dataset', 
                    perturbation_type='bernoulli', parameters={}, **kwargs):
        super().__init__(**kwargs)
        self.base_data = base_data
        self.dataset_name = dataset_name
        self.perturbation_type = perturbation_type
        self.parameters = parameters

    @property
    def configuration(self):
        return super().configuration | {
            'Based on' : self.base_data,
            'Dataset name' : self.dataset_name,
            'Type' : self.perturbation_type,
            'Parameters' : self.parameters,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        base_loader = get_data_loader(self.base_data, kwargs['data_loaders'])
        if len(base_loader) != 1:
            raise RuntimeError(f'Perturbing is only supported for single graph data.')
        data = PerturbationTransform(
            noise_type=self.perturbation_type,
            **self.parameters,
        )(base_loader.dataset[0])
        dataset = SingleGraphDataset(data)
        kwargs['data_loaders'][self.dataset_name] = DataLoader(dataset, batch_size=1, shuffle=False)
        return args, kwargs

@register_pipeline_member
class ExportData(OODSeparation):
    """ Pipeline member that exports datasets. 
    In `output_path`, tokens encapsulated in {} will be replaced by values from the dataset configuration"""
    
    name = 'ExportData'

    def __init__(self, datasets='all', ood_datasets=[dconstants.OOD_VAL, dconstants.OOD_TEST], output_path = './exported_datasets/{data.dataset}/{data.setting}-{data.ood_type}-{data.split_idx}.pkl', **kwargs):
        super().__init__(evaluate_on=ood_datasets, **kwargs)
        self.datasets = datasets
        self.output_path = output_path
        self.ood_datasets = ood_datasets
    
    @property
    def configuration(self):
        return super().configuration | {
            'Datasets' : f'{self.datasets}',
            'Output path' : self.output_path,
        }

    def __call__(self, *args, **kwargs):
        if self.datasets == 'all':
            dataset_names = set(kwargs['data_loaders'].keys())
        else:
            dataset_names = set(self.datasets)

        cfg: configuration.ExperimentConfiguration = kwargs['config']
        # Substitute tokens encapsulated in {} with
        output_path = self.output_path
        for match in set(re.findall(r'(\{.*?\})', self.output_path)):
            c = cfg
            path = match[1:-1].split('.')
            for prefix in path[:-1]:
                c = getattr(c, prefix)
            value = getattr(c, path[-1])
            output_path = output_path.replace(match, str(value))

        datasets = {name : kwargs['data_loaders'][name].dataset[0] for name in dataset_names}
        for name in self.ood_datasets:
            # Get vertices that are used for AUROC calculation in our experiments.
            self.evaluate_on = [name]
            auroc_labels, auroc_mask, _, _ = self.get_distribution_labels(mask=False, **kwargs)
            datasets[name].auroc_mask = auroc_mask
            datasets[name].is_in_distribution = auroc_labels
            # print(f'Created mask for ood dataset {name}')

        os.makedirs(osp.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb+') as f:
            pickle.dump({
                'data' : datasets,
                'configuration' : deepcopy(cfg.registry_configuration),
            }, f)

        return args, kwargs

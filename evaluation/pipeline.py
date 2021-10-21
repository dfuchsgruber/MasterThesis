import numpy as np
import torch
import evaluation.lipschitz
import plot.perturbations
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from model.density import GMMFeatureSpaceDensity
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from data.util import LabelMaskDataset

class EvaluateEmpircalLowerLipschitzBounds:
    """ Pipeline element for evaluation of Lipschitz bounds. """

    name = 'EvaluateEmpircalLowerLipschitzBounds'

    def __init__(self, num_perturbations_per_sample=5, min_perturbation=0.1, max_perturbation=5.0, num_perturbations = 10, seed=None, gpus=0):
        self.num_perturbations_per_sample = num_perturbations_per_sample
        self.min_perturbation = min_perturbation
        self.max_perturbation = max_perturbation
        self.num_perturbations = num_perturbations
        self.seed = None
        self.gpus = gpus

    def __call__(self, model, data_loader_train, data_loader_val, logger, **kwargs):
        assert len(data_loader_val) == 1, f'Empirical local lipschitz evaluation is currently only supported for semi-supervised tasks.'
        for dataset in data_loader_val:
            break # We just want the single dataset
    
        if self.gpus > 0:
            dataset = dataset.to('cuda') # A bit hacky, but short...7
            model = model.to('cuda')


        perturbations = evaluation.lipschitz.local_perturbations(model, dataset,
            perturbations=np.linspace(self.min_perturbation, self.max_perturbation, self.num_perturbations),
            num_perturbations_per_sample=self.num_perturbations_per_sample, seed=self.seed)
        pipeline_log(f'Created {self.num_perturbations_per_sample} perturbations in linspace({self.min_perturbation:.2f}, {self.max_perturbation:.2f}, {self.num_perturbations}) for validation samples.')
        smean, smedian, smax, smin = evaluation.lipschitz.local_lipschitz_bounds(perturbations)
        logger.log_metrics = {
            'slope_mean_perturbation' : smean,
            'slope_median_perturbation' : smedian,
            'slope_max_perturbation' : smax,
            'slope_min_perturbation' : smin,
        }
        # Plot the perturbations and log it
        fig, _ , _, _ = plot.perturbations.local_perturbations_plot(perturbations)
        if isinstance(logger, TensorBoardLogger):
            logger.experiment.add_figure('val_perturbations', fig)
        pipeline_log(f'Logged input vs. output perturbation plot.')
        
        return (model, data_loader_train, data_loader_val, logger), kwargs

class FitLogitDensity:
    """ Pipeline member that fits a density to the logit space of a model. """

    name = 'FitLogitDensity'

    def __init__(self, density_type='gmm', gpus=0):
        if density_type.lower() == 'gmm':
            self.density = GMMFeatureSpaceDensity()
            self.density_type = density_type.lower()
        else:
            raise RuntimeError(f'Unsupported logit space density {density_type}')
        self.gpus = gpus

    def __call__(self, model, data_loader_train, data_loader_val, logger, **kwargs):
        assert len(data_loader_val) == 1, f'Logit Space Density is currently only supported for semi-supervised tasks.'
        for data_train in data_loader_train:
            break # We just want the single dataset
        for data_val in data_loader_val:
            break # We just want the single dataset
    
        if self.gpus > 0:
            data_train = data_train.to('cuda')
            data_val = data_val.to('cuda')
            model = model.to('cuda')

        logits = model(data_train)[-1][data_train.mask]
        self.density.fit(logits, data_train.y[data_train.mask])
        if 'logit_density' in kwargs:
            pipeline_log('Density was already fit to logits, overwriting...')
        kwargs['logit_density'] = self.density
        pipeline_log(f'Fitted density of type {self.density_type} to training data logits.')
        return (model, data_loader_train, data_loader_val, logger), kwargs

class SelectClassLabels:
    """ Pipeline element that selects only a set of class labels for a given dataset. """

    name = 'SelectClassLabels'

    def __init__(self, select_labels=(0,), dataset='train'):
        self.select_labels = select_labels
        self.mode = dataset
    
    def __call__(self, model, data_loader_train, data_loader_val, logger, **kwargs):
        if self.mode == 'train':
            dataset = data_loader_train.dataset
            assert len(dataset) == 1, f'Select class labels is only supported for semi-supervised node classification.'
            data_loader_train = DataLoader(LabelMaskDataset(dataset, select_labels=self.select_labels), batch_size=1, shuffle=False)
        elif self.mode == 'val':
            dataset = data_loader_val.dataset
            assert len(dataset) == 1, f'Select class labels is only supported for semi-supervised node classification.'
            data_loader_val = DataLoader(LabelMaskDataset(dataset, select_labels=self.select_labels), batch_size=1, shuffle=False)

        num_train_unique = len(torch.unique(data_loader_train.dataset[0].y[data_loader_train.dataset[0].mask]))
        num_val_unique = len(torch.unique(data_loader_val.dataset[0].y[data_loader_val.dataset[0].mask]))

        data_loader_val(f'Selected class labels: Unique in train {num_train_unique}, unique in val {num_val_unique}.')
        return (model, data_loader_train, data_loader_val, logger), kwargs

class Pipeline:
    """ Pipeline for stuff to do after a model has been trained. """

    def __init__(self, members: list, config: dict, gpus=0):
        self.members = []
        for name in members:
            if name.lower() == EvaluateEmpircalLowerLipschitzBounds.name.lower():
                self.members.append(EvaluateEmpircalLowerLipschitzBounds(
                    num_perturbations=config['perturbations']['num'],
                    min_perturbation=config['perturbations']['min'],
                    max_perturbation=config['perturbations']['max'],
                    num_perturbations_per_sample=config['perturbations']['num_per_sample'],
                    seed=config['perturbations'].get('seed', None),
                    gpus=gpus,
                ))
            elif name.lower() == 'fitlogitdensitygmm':
                self.members.append(FitLogitDensity(
                    density_type='gmm',
                    gpus = gpus,
                ))
            elif name.lower() == 'selectclasslabelstrain':
                self.members.append(SelectClassLabels(
                    select_labels=config['select_class_labels_train'],
                    dataset='train',
                ))
            elif name.lower() == 'selectclasslabelsval':
                self.members.append(SelectClassLabels(
                    select_labels=config['select_class_labels_val'],
                    dataset='val',
                ))

            else:
                raise RuntimeError(f'Unrecognized evaluation pipeline member {name}')

    def __call__(self, *args, **kwargs):
        for member in self.members:
            args, kwargs = member(*args, **kwargs)
        return args, kwargs

def pipeline_log(string):
    print(f'EVALUATION PIPELINE - {string}')
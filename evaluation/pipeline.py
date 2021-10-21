import numpy as np
import torch
import evaluation.lipschitz
import plot.perturbations
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from model.density import GMMFeatureSpaceDensity
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader

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

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert len(kwargs['data_loader_val']) == 1, f'Empirical local lipschitz evaluation is currently only supported for semi-supervised tasks.'
        for dataset in kwargs['data_loader_val']:
            break # We just want the single dataset
    
        if self.gpus > 0:
            dataset = dataset.to('cuda') # A bit hacky, but short...7
            kwargs['model'] = kwargs['model'].to('cuda')

        perturbations = evaluation.lipschitz.local_perturbations(kwargs['model'], dataset,
            perturbations=np.linspace(self.min_perturbation, self.max_perturbation, self.num_perturbations),
            num_perturbations_per_sample=self.num_perturbations_per_sample, seed=self.seed)
        pipeline_log(f'Created {self.num_perturbations_per_sample} perturbations in linspace({self.min_perturbation:.2f}, {self.max_perturbation:.2f}, {self.num_perturbations}) for validation samples.')
        smean, smedian, smax, smin = evaluation.lipschitz.local_lipschitz_bounds(perturbations)
        kwargs['logger'].log_metrics({
            'slope_mean_perturbation' : smean,
            'slope_median_perturbation' : smedian,
            'slope_max_perturbation' : smax,
            'slope_min_perturbation' : smin,
        })
        # Plot the perturbations and log it
        fig, _ , _, _ = plot.perturbations.local_perturbations_plot(perturbations)
        if isinstance(kwargs['logger'], TensorBoardLogger):
            kwargs['logger'].experiment.add_figure('val_perturbations', fig)
        pipeline_log(f'Logged input vs. output perturbation plot.')
        
        return args, kwargs

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

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert len(kwargs['data_loader_train']) == 1 and len(kwargs['data_loader_val_all_classes']) == 1, f'Logit Space Density is currently only supported for semi-supervised tasks.'
        for data_train in kwargs['data_loader_train']:
            break # We just want the single dataset
    
        if self.gpus > 0:
            data_train = data_train.to('cuda')
            kwargs['model'] = kwargs['model'].to('cuda')

        logits = kwargs['model'](data_train)[-1][data_train.mask]
        self.density.fit(logits, data_train.y[data_train.mask])
        if 'logit_density' in kwargs:
            pipeline_log('Density was already fit to logits, overwriting...')
        kwargs['logit_density'] = self.density
        pipeline_log(f'Fitted density of type {self.density_type} to training data logits.')
        return args, kwargs

class EvaluateLogitDensity:
    """ 
    Pipeline member that evaluates the logit density at each sample in the validation set.
    It also logs histograms and statistics.
    """

    name = 'EvaluateLogitDensity'

    def __init__(self, gpus=0):
        self.gpus = gpus
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        for data_val in kwargs['data_loader_val_all_classes']:
            break # We just want the single dataset

        if self.gpus > 0:
            data_val = data_val.to('cuda')
            kwargs['model'] = kwargs['model'].to('cuda')

        logits = kwargs['model'](data_val)[-1][data_val.mask].cpu()
        density = kwargs['logit_density'].cpu()(logits)
        
        # Log histograms and metrics label-wise
        y = data_val.y[data_val.mask]
        for label in torch.unique(y):
            density_label = density[y == label]
            if isinstance(kwargs['logger'], TensorBoardLogger):
                kwargs['logger'].experiment.add_histogram(f'logit_density', density_label, global_step=label)
            kwargs['logger'].log_metrics({
                f'mean_logit_density' : density_label.mean(),
                f'std_logit_density' : density_label.std(),
                f'min_logit_density' : density_label.min(),
                f'max_logit_density' : density_label.max(),
                f'median_logit_density' : density_label.median(),
            }, step=label)
        pipeline_log(f'Evaluated logit density for entire validation dataset (labels : {torch.unique(y).cpu().tolist()}).')
        return args, kwargs

class LogLogits:
    """ Pipeline member that logs the logits of the validation data. """

    name = 'LogLogits'

    def __init__(self, gpus=0):
        self.gpus = gpus
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        for data_val in kwargs['data_loader_val_all_classes']:
            break # We just want the single dataset

        if self.gpus > 0:
            data_val = data_val.to('cuda')
            kwargs['model'] = kwargs['model'].to('cuda')

        logits = kwargs['model'](data_val)[-1][data_val.mask].cpu()
        if isinstance(kwargs['logger'], TensorBoardLogger):
            kwargs['logger'].experiment.add_embedding(logits, tag='val_logits')
        pipeline_log(f'Logged logits (size {logits.size()}) of entire validation dataset.')
    
        return args, kwargs


# class SelectClassLabels:
#     """ Pipeline element that selects only a set of class labels for a given dataset. """

#     name = 'SelectClassLabels'

#     def __init__(self, select_labels=(0,), dataset='train'):
#         self.select_labels = select_labels
#         self.mode = dataset
    
#     def __call__(self, model, data_loader_train, data_loader_val, logger, **kwargs):
#         if self.mode == 'train':
#             dataset = data_loader_train.dataset
#             assert len(dataset) == 1, f'Select class labels is only supported for semi-supervised node classification.'
#             data_loader_train = DataLoader(LabelMaskDataset(dataset, select_labels=self.select_labels), batch_size=1, shuffle=False)
#         elif self.mode == 'val':
#             dataset = data_loader_val.dataset
#             assert len(dataset) == 1, f'Select class labels is only supported for semi-supervised node classification.'
#             data_loader_val = DataLoader(LabelMaskDataset(dataset, select_labels=self.select_labels), batch_size=1, shuffle=False)

#         num_train_unique = len(torch.unique(data_loader_train.dataset[0].y[data_loader_train.dataset[0].mask]))
#         num_val_unique = len(torch.unique(data_loader_val.dataset[0].y[data_loader_val.dataset[0].mask]))

#         data_loader_val(f'Selected class labels: Unique in train {num_train_unique}, unique in val {num_val_unique}.')
#         return (model, data_loader_train, data_loader_val, logger), kwargs

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
            elif name.lower() == EvaluateLogitDensity.name.lower():
                self.members.append(EvaluateLogitDensity(
                    gpus=gpus,
                ))
            elif name.lower() == LogLogits.name.lower():
                self.members.append(LogLogits(
                    gpus=gpus,
                ))

            # elif name.lower() == 'selectclasslabelstrain':
            #     self.members.append(SelectClassLabels(
            #         select_labels=config['select_class_labels_train'],
            #         dataset='train',
            #     ))
            # elif name.lower() == 'selectclasslabelsval':
            #     self.members.append(SelectClassLabels(
            #         select_labels=config['select_class_labels_val'],
            #         dataset='val',
            #     ))

            else:
                raise RuntimeError(f'Unrecognized evaluation pipeline member {name}')

    def __call__(self, *args, **kwargs):
        for member in self.members:
            args, kwargs = member(*args, **kwargs)
        return args, kwargs

def pipeline_log(string):
    print(f'EVALUATION PIPELINE - {string}')
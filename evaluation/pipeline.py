import numpy as np
import torch
import evaluation.lipschitz
import plot.perturbations
from plot.density import plot_2d_log_density, plot_log_density_histograms
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from model.density import GMMFeatureSpaceDensity
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
import os.path as osp

def split_labels_into_id_and_ood(y, id_labels, ood_labels=None, id_label=0, ood_label=1):
    """ Creates new labels that correspond to in-distribution and out-ouf-distribution.
    
    Paramters:
    ----------
    y : torch.tensor, shape [N]
        Original class labels.
    id_labels : iterable
        All in-distribution labels.
    ood_labels : iterable
        All out-of-distribution labels or None. If None, it will be set to all labels in y that are not id.
    id_label : int
        The new label of id points.
    ood_label : int
        The new label of ood points.

    Returns:
    --------
    y_new : torch.tensor, shape [N]
        Labels corresponding to id and ood data.
    """
    if ood_labels is None:
        ood_labels = set(torch.unique(y).tolist()) - set(id_labels)

    y_new = torch.zeros_like(y)
    for label in id_labels:
        y_new[y == label] = id_label
    for label in ood_labels:
        y_new[y == label] = ood_label
    return y_new

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
        for data_val in kwargs['data_loader_val_all_classes']:
            break
    
        if self.gpus > 0:
            data_train = data_train.to('cuda')
            data_val = data_val.to('cuda')
            kwargs['model'] = kwargs['model'].to('cuda')

        logits = kwargs['model'](data_train)[-1][data_train.mask].cpu()
        logits_val = kwargs['model'](data_val)[-1][data_val.mask].cpu()

        torch.save({
            'logits' : logits,
            'labels' : data_train.y[data_train.mask].cpu(),
        }, osp.join(kwargs['artifact_directory'], 'logits_train.pt'))
        torch.save({
            'logits' : logits_val,
            'labels' : data_val.y[data_val.mask].cpu(),
        }, osp.join(kwargs['artifact_directory'], 'logits_val_all_classes.pt'))

        self.density.fit(logits, data_train.y[data_train.mask].cpu())
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

        logits = kwargs['model'](data_val)[-1][data_val.mask]
        density = kwargs['logit_density'](logits.cpu()).cpu()
        
        # Log histograms and metrics label-wise
        y = data_val.y[data_val.mask]
        for label in torch.unique(y):
            log_density_label = torch.log(density[y == label])
            if isinstance(kwargs['logger'], TensorBoardLogger):
                kwargs['logger'].experiment.add_histogram(f'logit_log_density', log_density_label, global_step=label)
            kwargs['logger'].log_metrics({
                f'mean_logit_log_density' : log_density_label.mean(),
                f'std_logit_log_density' : log_density_label.std(),
                f'min_logit_log_density' : log_density_label.min(),
                f'max_logit_log_density' : log_density_label.max(),
                f'median_logit_log_density' : log_density_label.median(),
            }, step=label)
        fig, ax = plot_log_density_histograms(torch.log(density.cpu()), y.cpu(), overlapping=False)
        fig.savefig(osp.join(kwargs['artifact_directory'], 'logit_log_density_histograms_all_classes.pdf'))
        pipeline_log(f'Evaluated logit density for entire validation dataset (labels : {torch.unique(y).cpu().tolist()}).')

        # Split into in-distribution and out-of-distribution
        labels_id_ood = split_labels_into_id_and_ood(y.cpu(), set(kwargs['config']['data']['train_labels']), id_label=0, ood_label=1)
        fig, ax = plot_log_density_histograms(torch.log(density.cpu()), labels_id_ood.cpu(), label_names={0 : 'id', 1 : 'ood'})
        fig.savefig(osp.join(kwargs['artifact_directory'], 'logit_log_density_histograms_id_vs_ood.pdf'))

        if logits.size()[1] == 2: # Plot logit densities in case of 2-d logit space
            fig, ax = plot_2d_log_density(logits.cpu(), labels_id_ood.cpu(), kwargs['logit_density'], label_names={0 : 'id', 1 : 'ood'})
            # if isinstance(kwargs['logger'], TensorBoardLogger):
            #     kwargs['logger'].experiment.add_figure('logit_density', fig)
            #     pipeline_log(f'Logged density plot for logit space.')
            fig.savefig(osp.join(kwargs['artifact_directory'], 'logit_log_density_contour.pdf'))

        return args, kwargs

class FitLogitSpacePCA:
    """ Fits PCA to the logit space using training and validation data. """

    name = 'FitLogitSpacePCA'

    def __init__(self, gpus=0):
        self.gpus = gpus

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        for data_train in kwargs['data_loader_train']:
            break # We just want the single dataset
        for data_val in kwargs['data_loader_val_all_classes']:
            break # We just want the single dataset

        if self.gpus > 0:
            data_train = data_train.to('cuda')
            data_val = data_val.to('cuda')
            kwargs['model'] = kwargs['model'].to('cuda')

        logits_train = kwargs['model'](data_train)[-1][data_train.mask].cpu().numpy()
        logits_val = kwargs['model'](data_val)[-1][data_val.mask].cpu().numpy()
        logits = np.concatenate([logits_train, logits_val], axis=0)

        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(logits)
        pipeline_log(f'Fit logit space PCA with {n_components} components. Explained variance ratio {pca.explained_variance_ratio_}')
        kwargs['logit_space_pca'] = pca

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
            kwargs['logger'].experiment.add_embedding(logits, tag='val_logits', metadata=data_val.y[data_val.mask].cpu().numpy())
        pipeline_log(f'Logged logits (size {logits.size()}) of entire validation dataset.')
    
        return args, kwargs

class Pipeline:
    """ Pipeline for stuff to do after a model has been trained. """

    def __init__(self, members: list, config: dict, gpus=0, ignore_exceptions=False):
        self.members = []
        self.ignore_exceptions = ignore_exceptions
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
            elif name.lower() == FitLogitSpacePCA.name.lower():
                self.members.append(FitLogitSpacePCA(
                    gpus=gpus,
                ))
            else:
                raise RuntimeError(f'Unrecognized evaluation pipeline member {name}')

    def __call__(self, *args, **kwargs):
        for member in self.members:
            try:
                args, kwargs = member(*args, **kwargs)
            except Exception as e:
                pipeline_log(f'{member.name} FAILED. Reason: "{e}"')
                if not self.ignore_exceptions:
                    raise e
        return args, kwargs

def pipeline_log(string):
    print(f'EVALUATION PIPELINE - {string}')
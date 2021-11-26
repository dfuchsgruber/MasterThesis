import numpy as np
import torch
import pytorch_lightning as pl
import evaluation.lipschitz
import plot.perturbations
import matplotlib.pyplot as plt
from evaluation.util import split_labels_into_id_and_ood, get_data_loader, run_model_on_datasets, count_neighbours_with_label, get_distribution_labels_leave_out_classes, separate_distributions_leave_out_classes
import evaluation.constants as evaluation_constants
from evaluation.callbacks import *
from plot.density import plot_2d_log_density
from plot.features import plot_2d_features
from plot.util import plot_histogram, plot_histograms, plot_2d_histogram
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from model.density import get_density_model
from model.dimensionality_reduction import DimensionalityReduction
from evaluation.logging import log_figure, log_histogram, log_embedding, log_metrics
from evaluation.features import inductive_feature_shift
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import os.path as osp
from data.util import label_binarize, data_get_summary, labels_in_dataset, vertex_intersection, labels_to_idx, graph_select_labels
from data.base import SingleGraphDataset
from data.transform import RemoveEdgesTransform
from warnings import warn
from itertools import product
from util import format_name, get_k_hop_neighbourhood
from copy import deepcopy
import traceback
import data.constants as data_constants
from torch_geometric.transforms import Compose

_ID_LABEL, _OOD_LABEL = 1, 0
FEATURE_SHIFT_EPS = 1e-10 # Zero feature shift is bad in log space

class PipelineMember:
    """ Superclass for all pipeline members. """

    def __init__(self, name=None, **kwargs):
        self._name = name

    @property
    def prefix(self):
        if self._name is None:
            return f''
        else:
            return f'{self._name}_'
    
    @property
    def suffix(self):
        if self._name is None:
            return f''
        else:
            return f'_{self._name}'

    @property
    def print_name(self):
        if self._name is None:
            return f'{self.name} (unnamed)'
        else:
            return f'{self.name} : "{self._name}"'
    
    @property
    def configuration(self):
        return {}

    def __str__(self):
        return '\n'.join([self.print_name] + [f'\t{key} : {value}' for key, value in self.configuration.items()])

class EvaluateEmpircalLowerLipschitzBounds(PipelineMember):
    """ Pipeline element for evaluation of Lipschitz bounds. """

    name = 'EvaluateEmpircalLowerLipschitzBounds'

    def __init__(self, num_perturbations_per_sample=5, min_perturbation=0.1, max_perturbation=5.0, num_perturbations = 10, 
            seed=None, gpus=0, evaluate_on=['val'], perturbation_type='noise', permute_per_sample=True, **kwargs):
        super().__init__(**kwargs)
        self.num_perturbations_per_sample = num_perturbations_per_sample
        self.min_perturbation = min_perturbation
        self.max_perturbation = max_perturbation
        self.num_perturbations = num_perturbations
        self.perturbation_type = perturbation_type
        self.permute_per_sample = permute_per_sample
        self.seed = seed
        self.gpus = gpus
        self.evaluate_on = evaluate_on

    @property
    def configuration(self):
        config = super().configuration
        config |= {
            'Perturbation type' : self.perturbation_type,
            'Number of perturbations per sample' : self.num_perturbations,
            'Seed' : self.seed,
            'Evaluate on' : self.evaluate_on,
        }
        if self.perturbation_type.lower() == 'noise':
            config['Min perturbation'] = self.min_perturbation
            config['Max perturbation'] = self.max_perturbation
        elif self.perturbation_type.lower() == 'derangement':
            config['Permutation per Sample'] = self.permute_per_sample
        return config

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        for name in self.evaluate_on:
            loader = get_data_loader(name, kwargs['data_loaders'])
            assert len(loader) == 1, f'Empirical local lipschitz evaluation is currently only supported for semi-supervised tasks.'
            for dataset in loader:
                break # Just want the single dataset
            if self.gpus > 0:
                dataset = dataset.to('cuda')
                kwargs['model'] = kwargs['model'].to('cuda')
            if self.perturbation_type.lower() == 'noise':
                perturbations = evaluation.lipschitz.local_perturbations(kwargs['model'], dataset,
                    perturbations=np.linspace(self.min_perturbation, self.max_perturbation, self.num_perturbations),
                    num_perturbations_per_sample=self.num_perturbations_per_sample, seed=self.seed)
                pipeline_log(f'Created {self.num_perturbations_per_sample} perturbations in linspace({self.min_perturbation:.2f}, {self.max_perturbation:.2f}, {self.num_perturbations}) for {name} samples.')
            elif self.perturbation_type.lower() == 'derangement':
                perturbations = evaluation.lipschitz.permutation_perturbations(kwargs['model'], dataset,
                    self.num_perturbations, num_perturbations_per_sample=self.num_perturbations_per_sample, 
                    seed=self.seed, per_sample=self.permute_per_sample)
                pipeline_log(f'Created {self.num_perturbations} permutations ({self.num_perturbations_per_sample} each) for {name} samples.')

            smean, smedian, smax, smin = evaluation.lipschitz.local_lipschitz_bounds(perturbations)
            log_metrics(kwargs['logs'], {
                f'{name}_slope_mean_perturbation' : smean,
                f'{name}_slope_median_perturbation' : smedian,
                f'{name}_slope_max_perturbation' : smax,
                f'{name}_slope_min_perturbation' : smin,
            }, f'empirical_lipschitz{self.suffix}')
            # Plot the perturbations and log it
            fig, _ , _, _ = plot.perturbations.local_perturbations_plot(perturbations)
            log_figure(kwargs['logs'], fig, f'{name}_perturbations', f'empirical_lipschitz{self.suffix}', save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Logged input vs. output perturbation plot for dataset {name}.')
            plt.close(fig)
        
        return args, kwargs

class OODSeparation(PipelineMember):
    """ Base class to perform any kind of method that separates id from ood data. """

    name = 'OODSeparation'
    
    def __init__(self, *args, separate_distributions_by='train_label', separate_distributions_tolerance=0.0,
            evaluate_on=[data_constants.VAL], kind='leave-out-classes', **kwargs):
        super().__init__(*args, **kwargs)
        self.separate_distributions_by = separate_distributions_by
        self.separate_distributions_tolerance = separate_distributions_tolerance
        self.kind = kind
        self.evaluate_on = evaluate_on

    @property
    def configuration(self):
        return super().configuration | {
            'Evaluate on' : self.evaluate_on,
            'Separate distributions by' : self.separate_distributions_by,
            'Separate distributions tolerance' : self.separate_distributions_tolerance,
            'OOD kind' : self.kind,
        }

    def _get_distribution_labels_leave_out_classes(self, **kwargs):
        """ Gets labels for id vs ood where ood data is left out classes.
        
        Returns:
        --------
        auroc_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for auroc calculation.
        auroc_mask : torch.Tensor, shape [N]
            Which samples should be used for AUROC calculation.
        distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions.
        distribution_label_names : dict
            Mapping that names all the labels in `distribution_labels`.
        """
        # Classify vertices according to which distribution they belong to
        k = len(kwargs['config']['model']['hidden_sizes']) + 1
        distribution_labels = get_distribution_labels_leave_out_classes(
            [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on],
            k, # Receptive field of each vertex is num-hidden-layers + 1
            kwargs['config']['data']['train_labels'],
            mask = True,
            threshold = self.separate_distributions_tolerance,
        )
        auroc_labels, auroc_mask = separate_distributions_leave_out_classes(distribution_labels, self.separate_distributions_by)
        distribution_label_names = {
            evaluation_constants.OOD_CLASS_NO_ID_CLASS_NBS : f'ood, no id nbs in {k} hops', 
            evaluation_constants.OOD_CLASS_ID_CLASS_NBS : f'ood, id nbs in {k}-hops',
            evaluation_constants.ID_CLASS_NO_OOD_CLASS_NBS : f'id, no ood nbs in {k}-hops', 
            evaluation_constants.ID_CLASS_ODD_CLASS_NBS : f'id, ood nbs in {k}-hops'
        }
        return auroc_labels, auroc_mask, distribution_labels, distribution_label_names

    def get_distribution_labels(self, **kwargs):
        """ Gets labels for id vs ood where ood data is left out classes.

        Parameters:
        -----------
        kind : str
            With which method to get out of distribution data.
        
        Returns:
        --------
        auroc_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for auroc calculation.
        auroc_mask : torch.Tensor, shape [N]
            Which samples should be used for AUROC calculation.
        distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions.
        distribution_label_names : dict
            Mapping that names all the labels in `distribution_labels`.
        """
        if self.kind.lower() in ('leave-out-classes', 'loc', 'leave_out_classes'):
            return self._get_distribution_labels_leave_out_classes(**kwargs)
        else:
            raise RuntimeError(f'Could not separate distribution labels (id vs ood) by unknown type {self.kind}.')

class OODDetection(OODSeparation):
    """ Pipeline member to perform OOD detection for a given metric. Evaluates AUROC scores and logs plots. """

    name = 'OODDetection'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def ood_detection(self, proxy, labels, proxy_name, auroc_labels, auroc_mask, distribution_labels, distribution_label_names,
                        plot_proxy_log_scale=True, log_plots=True,**kwargs):
        """ Performs ood detection and logs metrics and plots.
        
        Parameters:
        -----------
        proxy : torch.Tensor, shape [N]
            The proxy for separating id and ood. Higher values should be assigned to id data.
        labels : torch.Tensor, shape [N]
            Ground truth labels. Used to separate the proxy values by ground truth label.
        proxy_name : str
            Name of the proxy to use.
        auroc_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for auroc calculation.
        auroc_mask : torch.Tensor, shape [N]
            Which samples should be used for AUROC calculation.
        distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions.
        distribution_label_names : dict
            Mapping that names all the labels in `distribution_labels`.
        plot_proxy_log_scale : bool
            If `True`, the proxy will be plotted in log scale.
        log_plots : bool
            If `True`, plots will be logged.
        """

        # Log histograms and metrics label-wise
        if log_plots:
            y = labels.cpu()
            for label in torch.unique(y):
                proxy_label = proxy[y == label]
                log_histogram(kwargs['logs'], proxy_label.cpu().numpy(), f'{proxy_name}', global_step=label, label_suffix=str(label.item()))
                log_metrics(kwargs['logs'], {
                    f'{self.prefix}mean_{proxy_name}' : proxy_label.mean(),
                    f'{self.prefix}std_{proxy_name}' : proxy_label.std(),
                    f'{self.prefix}min_{proxy_name}' : proxy_label.min(),
                    f'{self.prefix}max_{proxy_name}' : proxy_label.max(),
                    f'{self.prefix}median_{proxy_name}' : proxy_label.median(),
                }, f'{proxy_name}', step=label)
            fig, ax = plot_histograms(proxy.cpu(), y.cpu(), log_scale=plot_proxy_log_scale, kind='vertical', x_label='proxy', y_label='Class')
            log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_all_classes{self.suffix}', f'{proxy_name}_plots', save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Evaluated {proxy_name}.')
            plt.close(fig)

        if log_plots:
            fig, ax = plot_histograms(proxy.cpu(), distribution_labels.cpu(), 
                label_names=distribution_label_names,
                kind='vertical', kde=True, log_scale=plot_proxy_log_scale,  x_label=proxy_name, y_label='Kind')
            log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_all_kinds{self.suffix}', f'{proxy_name}_plots', save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Saved {proxy_name} (all kinds) histogram to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_histograms_all_kinds{self.suffix}.pdf')))
            plt.close(fig)

            fig, ax = plot_histograms(proxy[auroc_mask].cpu(), auroc_labels[auroc_mask].cpu().long(), 
                label_names={0 : 'Out ouf distribution', 1 : 'In distribution'},
                kind='overlapping', kde=True, log_scale=plot_proxy_log_scale,  x_label='Softmax-proxy', y_label='Kind')
            log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_id_vs_ood{self.suffix}', f'{proxy_name}_plots', save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Saved {proxy_name} histogram (id vs ood) to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_histograms_id_vs_ood{self.suffix}.pdf')))
            plt.close(fig)

        # Calculate area under the ROC for separating in-distribution (label 1) from out of distribution (label 0)
        roc_auc = roc_auc_score(auroc_labels[auroc_mask].cpu().long().numpy(), proxy[auroc_mask].cpu().numpy()) # higher proxy -> higher uncertainty
        kwargs['metrics'][f'auroc_{proxy_name}{self.suffix}'] = roc_auc
        log_metrics(kwargs['logs'], {f'auroc_{proxy_name}{self.suffix}' : roc_auc}, f'{proxy_name}_plots')
        
class EvaluateLogitEnergy(OODDetection):
    """ Pipeline member to evaluate the Logit Energy curves of the model for in-distribution and out-of-distribution data. """

    name = 'EvaluateLogitEnergy'

    def __init__(self, gpus=0, evaluate_on=[data_constants.VAL], separate_distributions_by='train-label', 
                separate_distributions_tolerance=0.0, model_kwargs={}, log_plots=True, **kwargs):
        super().__init__(separate_distributions_by=separate_distributions_by, 
                            separate_distributions_tolerance=separate_distributions_tolerance,
                            evaluate_on=evaluate_on,
                            **kwargs)
        self.gpus = gpus
        self.model_kwargs = model_kwargs
        self.log_plots = log_plots

    @property
    def configuration(self):
        return super().configuration | {
            'Kwargs for model' : self.model_kwargs,
            'Log plots' : self.log_plots,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        data_loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
        energy, labels = run_model_on_datasets(kwargs['model'], data_loaders, callbacks=[
                make_callback_get_softmax_energy(mask=True),
                make_callback_get_ground_truth(mask=True),
            ], gpus=self.gpus, model_kwargs=self.model_kwargs)
        energy, labels = torch.cat(energy), torch.cat(labels)

        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)
        self.ood_detection(-energy, labels, 'logit-energy', auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
        
        return args, kwargs

class EvaluateSoftmaxEntropy(OODDetection):
    """ Pipeline member to evaluate the Softmax Entropy curves of the model for in-distribution and out-of-distribution data. """

    name = 'EvaluateSoftmaxEntropy'

    def __init__(self, gpus=0, evaluate_on=[data_constants.VAL], separate_distributions_by='train-label', 
                separate_distributions_tolerance=0.0, model_kwargs={}, log_plots=True, **kwargs):
        super().__init__(separate_distributions_by=separate_distributions_by, 
                            separate_distributions_tolerance=separate_distributions_tolerance,
                            evaluate_on=evaluate_on,
                            **kwargs)
        self.gpus = gpus
        self.model_kwargs = model_kwargs
        self.log_plots = log_plots

    @property
    def configuration(self):
        return super().configuration | {
            'Kwargs for model' : self.model_kwargs,
            'Log plots' : self.log_plots,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        data_loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
        entropy, labels = run_model_on_datasets(kwargs['model'], data_loaders, callbacks=[
                make_callback_get_softmax_entropy(mask=True),
                make_callback_get_ground_truth(mask=True),
            ], gpus=self.gpus, model_kwargs=self.model_kwargs)
        entropy, labels = torch.cat(entropy), torch.cat(labels)

        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)
        self.ood_detection(-entropy, labels, 'softmax-entropy', auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
        
        return args, kwargs

class FeatureDensity(OODDetection):
    """ Superclass for pipeline members that fit a feature density. """

    name = 'FeatureDensity'

    def __init__(self, gpus=0, fit_to=[data_constants.TRAIN], 
        fit_to_ground_truth_labels=[data_constants.TRAIN], evaluate_on=[data_constants.TEST], 
        model_kwargs_fit={}, model_kwargs_evaluate={}, separate_distributions_by='train_label',
        separate_distributions_tolerance=0.0, kind='leave_out_classes',
        **kwargs):
        super().__init__(
            separate_distributions_by=separate_distributions_by,
            separate_distributions_tolerance=separate_distributions_tolerance,
            kind=kind,
            evaluate_on=evaluate_on,
            **kwargs
        )
        self.gpus = gpus
        self.fit_to = fit_to
        self.fit_to_ground_truth_labels = fit_to_ground_truth_labels
        self.model_kwargs_fit = model_kwargs_fit
        self.model_kwargs_evaluate = model_kwargs_evaluate

    @property
    def configuration(self):
        return super().configuration | {
            'Fit to' : self.fit_to,
            'Use ground truth labels for fit on' : self.fit_to_ground_truth_labels,
            'Kwargs for model (fit)' : self.model_kwargs_fit,
            'Kwargs for model (evaluate)' : self.model_kwargs_evaluate,
        }

    def _get_features_and_labels_to_fit(self, **kwargs):
        features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], gpus=self.gpus, model_kwargs=self.model_kwargs_fit)
        for idx, name in enumerate(self.fit_to):
            if name.lower() in self.fit_to_ground_truth_labels: # Override predictions with ground truth for training data
                predictions[idx] = label_binarize(labels[idx], num_classes=predictions[idx].size(1)).float()
        return torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)

    def _get_features_and_labels_to_evaluate(self, **kwargs):
        features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate)
        return torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)

class FitFeatureDensityGrid(FeatureDensity):
    """ Pipeline member that fits a grid of several densities to the feature space of a model. """

    name = 'FitFeatureDensityGrid'

    def __init__(self, fit_to=[data_constants.TRAIN], fit_to_ground_truth_labels=[data_constants.TRAIN], 
                    evaluate_on=[data_constants.VAL], density_types={}, dimensionality_reductions={}, log_plots=False, gpus=0,
                    separate_distributions_by='train-label', separate_distributions_tolerance=0.0, seed=1337,
                    **kwargs):
        super().__init__(gpus=gpus, fit_to=fit_to, fit_to_ground_truth_labels=fit_to_ground_truth_labels, 
                        separate_distributions_by=separate_distributions_by,
                        separate_distributions_tolerance=separate_distributions_tolerance,
                        evaluate_on=evaluate_on, **kwargs)
        self.density_types = density_types
        self.dimensionality_reductions = dimensionality_reductions
        self.log_plots = log_plots
        self.seed = seed

    @property
    def configuration(self):
        return super().configuration | {
            'Density types' : self.density_types,
            'Dimensionality Reductions' : self.dimensionality_reductions,
            'Log plots' : self.log_plots,
            'Seed' : self.seed,
        }
        
    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        if self.seed is not None:
            pl.seed_everything(self.seed)

        # Only compute data once
        features_to_fit, predictions_to_fit, labels_to_fit = self._get_features_and_labels_to_fit(**kwargs)

        # Note that for `self.fit_to_ground_truth_labels` data, the `predictions_to_fit` are overriden with a 1-hot ground truth
        features_to_evaluate, predictions_to_evaluate, labels_to_evaluate = self._get_features_and_labels_to_evaluate(**kwargs)
        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)

        # Grid over dimensionality reductions
        for dim_reduction_type, dim_reduction_grid in self.dimensionality_reductions.items():
            keys = list(dim_reduction_grid.keys())
            for values in product(*[dim_reduction_grid[k] for k in keys]):
                dim_reduction_config = {key : values[idx] for idx, key in enumerate(keys)}
                dim_reduction = DimensionalityReduction(type=dim_reduction_type, per_class=False, **dim_reduction_config)
                dim_reduction.fit(features_to_fit, predictions_to_fit)
                pipeline_log(f'{self.name} fitted dimensionality reduction {dim_reduction.compressed_name}')

                # TODO: dim-reductions transform per-class, but this is not supported here, so we just take any and set `per_class` to false in its constructor
                features_to_fit_reduced = torch.from_numpy(list(dim_reduction.transform(features_to_fit).values())[0])
                features_to_evaluate_reduced = torch.from_numpy(list(dim_reduction.transform(features_to_evaluate).values())[0])

                # Grid over feature space densities
                for density_type, density_grid in self.density_types.items():
                    keys_density = list(density_grid.keys())
                    for values_density in product(*[density_grid[k] for k in keys_density]):
                        density_config = {key : values_density[idx] for idx, key in enumerate(keys_density)}
                        density_model = get_density_model(
                            density_type=density_type, 
                            dimensionality_reduction={'type' : None}, # Features are already reduced
                            **density_config,
                            )
                        density_model.fit(features_to_fit_reduced, predictions_to_fit)
                        log_density = density_model(features_to_evaluate_reduced).cpu()
                        pipeline_log(f'{self.name} fitted density {density_model.compressed_name}')                        

                        self.ood_detection(log_density, labels_to_evaluate,
                             f'{density_model.compressed_name}:{dim_reduction.compressed_name}',
                             auroc_labels, auroc_mask, distribution_labels, distribution_label_names, 
                             plot_proxy_log_scale=True, log_plots=self.log_plots, **kwargs
                            )

        return args, kwargs

class FitFeatureSpacePCAIDvsOOD(OODSeparation):
    """ Fits a 2d PCA to the feature space and separates id and ood data. """

    name = 'FitFeatureSpacePCAIDvsOOD'

    def __init__(self, gpus=0, fit_to=[data_constants.TRAIN], **kwargs):
            super().__init__(**kwargs)
            self.gpus = gpus
            self.fit_to = fit_to

    @property
    def configuration(self):
        return super().configuration | {
            'Fit to' : self.fit_to,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], gpus=self.gpus)
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(features.cpu().numpy())
        fig, ax = plot_2d_features(torch.tensor(transformed), labels)
        log_figure(kwargs['logs'], fig, f'pca_2d_id_vs_ood{self.suffix}_visualization_data_fit', 'feature_space_pca_id_vs_ood', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Logged 2d pca fitted to {self.fit_to}')
        plt.close(fig)

        features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], gpus=self.gpus)
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)
        transformed = pca.transform(features.cpu().numpy())
        fig, ax = plot_2d_features(torch.tensor(transformed), labels)
        log_figure(kwargs['logs'], fig, f'pca_2d_id_vs_ood{self.suffix}_visualization_by_label', 'feature_space_pca_id_vs_ood', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Logged 2d pca fitted to {self.fit_to}, evaluated on {self.evaluate_on} by label')
        plt.close(fig)

        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)
        fig, ax = plot_2d_features(torch.tensor(transformed), distribution_labels, distribution_label_names)
        log_figure(kwargs['logs'], fig, f'pca_2d_id_vs_ood{self.suffix}_visualization_by_id_vs_ood', 'feature_space_pca_id_vs_ood', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Logged 2d pca fitted to {self.fit_to}, evaluated on {self.evaluate_on} by in-distribution vs out-of-distribution')
        plt.close(fig)

        return args, kwargs

class FitFeatureSpacePCA(PipelineMember):
    """ Fits PCA to the feature space using training and validation data. """

    name = 'FitFeatureSpacePCA'

    def __init__(self, gpus=0, fit_to=['train'], evaluate_on=[], num_components=16, **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.fit_to = fit_to
        self.evaluate_on = evaluate_on
        self.num_components = num_components

    @property
    def configuration(self):
        return super().configuration | {
            'Fit to' : self.fit_to,
            'Evaluate on' : self.evaluate_on,
            'Number of components' : self.num_components,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], gpus=self.gpus)
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)

        pca = PCA(n_components=self.num_components)
        projected = pca.fit_transform(features.cpu().numpy())

        log_embedding(kwargs['logs'], projected, f'pca_{self.suffix}', labels.cpu().numpy(), save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Fit feature space PCA with {self.num_components} components. Explained variance ratio {pca.explained_variance_ratio_}')

        if len(self.evaluate_on) > 0 and self.num_components > 2:
            warn(f'Attempting to evalute PCA on {self.evaluate_on} but dimension is {self.num_components} != 2. No plots created.')

        if self.num_components == 2:
            loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
            features, predictions, labels = run_model_on_datasets(kwargs['model'], loaders, gpus=self.gpus)
            for idx, data_name in enumerate(self.evaluate_on):

                projected = pca.fit_transform(features[idx].cpu().numpy())
                fig, ax = plot_2d_features(torch.tensor(projected), labels[idx])
                log_figure(kwargs['logs'], fig, f'pca_{self.suffix}_{data_name}_gnd', 'pca', save_artifact=kwargs['artifact_directory'])
                plt.close(fig)

                fig, ax = plot_2d_features(torch.tensor(projected), predictions[idx].argmax(dim=1))
                log_figure(kwargs['logs'], fig, f'pca_{self.suffix}_{data_name}_predicted', 'pca', save_artifact=kwargs['artifact_directory'])
                plt.close(fig)

        kwargs['feature_space_pca'] = pca

        return args, kwargs

class LogFeatures(PipelineMember):
    """ Pipeline member that logs the features of the validation data. """

    name = 'LogFeatures'

    def __init__(self, gpus=0, evaluate_on=['val'], **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.evaluate_on = evaluate_on

    @property
    def configuration(self):
        return super().configuration | {
            'Evaluate on' : self.evaluate_on,
        }
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        features_all, predictions_all, labels_all = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], gpus=self.gpus)
        for name, predictions, features, labels in zip(self.evaluate_on, features_all, predictions_all, labels_all):
            log_embedding(kwargs['logs'], features.cpu().numpy(), f'{name}_features', labels.cpu().numpy(), save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Logged features (size {features.size()}) of dataset {name}')
            
        return args, kwargs

class PrintDatasetSummary(PipelineMember):
    """ Pipeline member that prints dataset statistics. """

    name = 'PrintDatasetSummary'

    def __init__(self, gpus=0, evaluate_on=['train', 'val'], **kwargs):
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

class LogInductiveFeatureShift(PipelineMember):
    """ Logs the feature shift in vertices of the train-graph when re-introducing new edges of the val-graph. """

    name = 'LogInductiveFeatureShift'

    def __init__(self, gpus=0, data_before=data_constants.TRAIN, data_after=data_constants.VAL, **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.data_before = data_before
        self.data_after = data_after

    @property
    def configuration(self):
        return super().configuration | {
            'Dataset before shift' : {self.data_before},
            'Dataset after shift' : {self.data_after},
        }
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        
        receptive_field_size = len(kwargs['config']['model']['hidden_sizes']) + 1
        callbacks = [
            make_callback_get_features(mask=False), 
            make_callback_get_data(),
        ]
        for k in range(1, receptive_field_size + 1):
            callbacks += [
                make_callback_count_neighbours_with_labels(kwargs['config']['data']['train_labels'], k, mask=False),
                make_callback_count_neighbours(k, mask=False),
            ]
        results = run_model_on_datasets(kwargs['model'], [get_data_loader(data, kwargs['data_loaders']) for data in (self.data_before, self.data_after)], gpus=self.gpus, callbacks=callbacks)
        features, data, num_nbs_in_train_labels, num_nbs = results[0], results[1], results[2::2], results[3::2]

        idx_before, idx_after = vertex_intersection(data[0], data[1])
        shift = (features[0][idx_before] - features[1][idx_after]).norm(dim=1)
        
        # Log the feature shift by "in mask / not in mask"
        fig, ax = plot_histograms(shift.cpu() + FEATURE_SHIFT_EPS, data[0].mask[idx_before].cpu(), 
                label_names={True : f'In {self.data_before}', False : f'Not in {self.data_before}'}, log_scale=True, kind='overlapping', x_label='Feature Shift')
        log_figure(kwargs['logs'], fig, f'feature_shift_by_mask', f'inductive_feature_shift{self.suffix}', save_artifact=kwargs['artifact_directory'])
        plt.close(fig)
        pipeline_log(f'Logged inductive feature shift for data {self.data_before} -> {self.data_after} by mask.')

        for k in range(1, receptive_field_size + 1):
            fraction = 1 - (num_nbs_in_train_labels[k - 1][1].float() / (num_nbs[k - 1][1] + 1e-12))
            fig, ax = plot_2d_histogram(shift.cpu() + FEATURE_SHIFT_EPS, fraction[idx_after], x_label='Log Feature Shift', y_label=f'Fraction of ood vertices in {k} neighbourhood', log_scale_x=True)
            log_figure(kwargs['logs'], fig, f'feature_shift_by_{k}_nbs', f'inductive_feature_shift{self.suffix}', save_artifact=kwargs['artifact_directory'])
            plt.close(fig)
            pipeline_log(f'Logged inductive feature shift for data {self.data_before} -> {self.data_after} by {k}-hop neighbourhood.')
        
        return args, kwargs

class LogInductiveSoftmaxEntropyShift(PipelineMember):
    """ Logs the shift of softmax entropy in vertices of the train-graph when re-introducing new edges of the val-graph. """

    name = 'LogInductiveSoftmaxEntropyShift'

    def __init__(self, gpus=0, data_before=data_constants.TRAIN, data_after=data_constants.VAL, **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.data_before = data_before
        self.data_after = data_after

    @property
    def configuration(self):
        return super().configuration | {
            'Dataset before shift' : {self.data_before},
            'Dataset after shift' : {self.data_after},
        }
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        
        receptive_field_size = len(kwargs['config']['model']['hidden_sizes']) + 1
        callbacks = [
            make_callback_get_softmax_entropy(mask=False), 
            make_callback_get_data(),
        ]
        for k in range(1, receptive_field_size + 1):
            callbacks += [
                make_callback_count_neighbours_with_labels(kwargs['config']['data']['train_labels'], k, mask=False),
                make_callback_count_neighbours(k, mask=False),
            ]
        results = run_model_on_datasets(kwargs['model'], [get_data_loader(data, kwargs['data_loaders']) for data in (self.data_before, self.data_after)], gpus=self.gpus, callbacks=callbacks)
        entropy, data, num_nbs_in_train_labels, num_nbs = results[0], results[1], results[2::2], results[3::2]

        idx_before, idx_after = vertex_intersection(data[0], data[1])
        shift = -(entropy[0][idx_before] - entropy[1][idx_after])
        
        # Log the entropy shift by "in mask / not in mask"
        fig, ax = plot_histograms(shift.cpu() + 0, data[0].mask[idx_before].cpu(), 
                label_names={True : f'In {self.data_before}', False : f'Not in {self.data_before}'}, log_scale=False, kind='overlapping', x_label='Entropy Shift')
        log_figure(kwargs['logs'], fig, f'entropy_shift_by_mask', f'inductive_entropy_shift{self.suffix}', save_artifact=kwargs['artifact_directory'])
        plt.close(fig)
        pipeline_log(f'Logged inductive entropy shift for data {self.data_before} -> {self.data_after} by mask.')

        for k in range(1, receptive_field_size + 1):
            fraction = 1 - (num_nbs_in_train_labels[k - 1][1].float() / (num_nbs[k - 1][1] + 1e-12))
            fig, ax = plot_2d_histogram(shift.cpu() + FEATURE_SHIFT_EPS, fraction[idx_after], x_label='Entropy Shift', y_label=f'Fraction of ood vertices in {k} neighbourhood', log_scale_x=False)
            log_figure(kwargs['logs'], fig, f'entropy_shift_by_{k}_nbs', f'inductive_entropy_shift{self.suffix}', save_artifact=kwargs['artifact_directory'])
            plt.close(fig)
            pipeline_log(f'Logged inductive entropy shift for data {self.data_before} -> {self.data_after} by {k}-hop neighbourhood.')
        
        return args, kwargs

class SubsetDataByLabel(PipelineMember):
    """ Creates a new data loader that holds a subset of some dataset with only certain labels. """

    name = 'SubsetDataByLabel'

    def __init__(self, base_data=data_constants.VAL, subset_name='unnamed-subset', labels='all', **kwargs):
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

pipeline_members = [
    EvaluateEmpircalLowerLipschitzBounds,
    LogFeatures,
    FitFeatureSpacePCA,
    FitFeatureSpacePCAIDvsOOD,
    PrintDatasetSummary,
    LogInductiveFeatureShift,
    LogInductiveSoftmaxEntropyShift,
    EvaluateLogitEnergy,
    EvaluateSoftmaxEntropy,
    FitFeatureDensityGrid,
    SubsetDataByLabel,
]


def pipeline_configs_from_grid(config, delimiter=':'):
    """ Builds a list of pipeline configs from a grid configuration. 
    
    Parameters:
    -----------
    config : dict
        The pipeline member config.
    
    Returns:
    --------
    configs : list
        A list of pipeline member configs from the grid or just the single config, if no grid was specified.
    """
    config = dict(config).copy() # To avoid errors?
    grid = config.pop('pipeline_grid', None)
    if grid is None:
        if 'name' in config:
            config['name'] = format_name(config['name'], config.get('name_args', []), config)
        return [config]
    else:
        parameters = grid # dict of lists
        keys = list(parameters.keys())
        configs = []
        for values in product(*[parameters[k] for k in keys]):
            subconfig = deepcopy(config)
            for idx, key in enumerate(keys):
                path = key.split(delimiter)
                target = subconfig
                for x in path[:-1]:
                    target = target[x]
                target[path[-1]] = values[idx]
            if 'name' in subconfig:
                subconfig['name'] = format_name(subconfig['name'], subconfig.get('name_args', []), subconfig, delimiter=delimiter)
            configs.append(subconfig)
        return configs

class Pipeline:
    """ Pipeline for stuff to do after a model has been trained. """

    def __init__(self, members: list, config: dict, gpus=0, ignore_exceptions=False):
        self.members = []
        self.ignore_exceptions = ignore_exceptions
        self.config = config
        idx = 0
        for idx, entry in enumerate(members):
            configs = pipeline_configs_from_grid(entry)
            for member in configs:
                for _class in pipeline_members:
                    if member['type'].lower() == _class.name.lower():
                        self.members.append(_class(
                            gpus = gpus,
                            pipeline_idx = idx,
                            **member
                        ))
                        idx += 1
                        break
                else:
                    raise RuntimeError(f'Unrecognized evaluation pipeline member {member}')

    def __call__(self, *args, **kwargs):
        for member in self.members:
            try:
                args, kwargs = member(*args, **kwargs)
            except Exception as e:
                pipeline_log(f'{member.print_name} FAILED. Reason: "{e}"')
                print(traceback.format_exc())
                if not self.ignore_exceptions:
                    raise e
        return args, kwargs

    def __str__(self):
        return '\n'.join([
            'Evaluation Pipeline',
        ] + [
            f'{member}' for member in self.members
        ])

def pipeline_log(string):
    print(f'EVALUATION PIPELINE - {string}')
import numpy as np
import torch
import pytorch_lightning as pl
import evaluation.lipschitz
import plot.perturbations
import matplotlib.pyplot as plt
from evaluation.util import split_labels_into_id_and_ood, get_data_loader, run_model_on_datasets, count_neighbours_with_label, get_distribution_labels_leave_out_classes, separate_distributions_leave_out_classes, get_distribution_labels_perturbations
import evaluation.constants as evaluation_constants
from evaluation.callbacks import *
from plot.density import plot_2d_log_density, plot_density
from plot.features import plot_2d_features
from plot.util import plot_histogram, plot_histograms, plot_2d_histogram
from plot.calibration import plot_calibration
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from model.density import get_density_model
from model.dimensionality_reduction import DimensionalityReduction
from evaluation.logging import log_figure, log_histogram, log_embedding, log_metrics
from evaluation.features import inductive_feature_shift
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
import os.path as osp
from data.util import label_binarize, data_get_summary, labels_in_dataset, vertex_intersection, labels_to_idx, graph_select_labels
from data.base import SingleGraphDataset
from data.transform import RemoveEdgesTransform, PerturbationTransform
from warnings import warn
from itertools import product
from util import format_name, get_k_hop_neighbourhood
from copy import deepcopy
import traceback
import data.constants as data_constants
from torch_geometric.transforms import Compose
from metrics import expected_calibration_error

_ID_LABEL, _OOD_LABEL = 1, 0
FEATURE_SHIFT_EPS = 1e-10 # Zero feature shift is bad in log space
ENTROPY_EPS = 1e-20 # For entropy of zero classes
VAR_EPS = 1e-12
ECE_EPS = 1e-12

class PipelineMember:
    """ Superclass for all pipeline members. """

    def __init__(self, name=None, log_plots=True, model_kwargs={}, model_kwargs_fit=None, model_kwargs_evaluate=None, **kwargs):
        self._name = name
        self.model_kwargs = model_kwargs
        if model_kwargs_fit is None:
            model_kwargs_fit = self.model_kwargs
        if model_kwargs_evaluate is None:
            model_kwargs_evaluate = self.model_kwargs
        self.model_kwargs_fit = model_kwargs_fit
        self.model_kwargs_evaluate = model_kwargs_evaluate
        self.log_plots = log_plots

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
        config = {
            'Kwargs to model calls' : self.model_kwargs,
            'Log plots' : self.log_plots,
        }
        if self.model_kwargs_fit != self.model_kwargs:
            config['Kwargs to model calls (fit)'] = self.model_kwargs_fit
        if self.model_kwargs_evaluate != self.model_kwargs:
            config['Kwargs to model calls (evaluate)'] = self.model_kwargs_evaluate
        return config

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
                    num_perturbations_per_sample=self.num_perturbations_per_sample, seed=self.seed, model_kwargs=self.model_kwargs)
                pipeline_log(f'Created {self.num_perturbations_per_sample} perturbations in linspace({self.min_perturbation:.2f}, {self.max_perturbation:.2f}, {self.num_perturbations}) for {name} samples.')
            elif self.perturbation_type.lower() == 'derangement':
                perturbations = evaluation.lipschitz.permutation_perturbations(kwargs['model'], dataset,
                    self.num_perturbations, num_perturbations_per_sample=self.num_perturbations_per_sample, 
                    seed=self.seed, per_sample=self.permute_per_sample, model_kwargs=self.model_kwargs)
                pipeline_log(f'Created {self.num_perturbations} permutations ({self.num_perturbations_per_sample} each) for {name} samples.')

            smean, smedian, smax, smin = evaluation.lipschitz.local_lipschitz_bounds(perturbations)
            metrics = {
                f'{name}_slope_mean_perturbation' : smean,
                f'{name}_slope_median_perturbation' : smedian,
                f'{name}_slope_max_perturbation' : smax,
                f'{name}_slope_min_perturbation' : smin,
            }
            log_metrics(kwargs['logs'], metrics, f'empirical_lipschitz{self.suffix}')
            for metric, value in metrics.items():
                kwargs['metrics'][f'empirical_lipschitz_{metric}{self.suffix}'] = value
            # Plot the perturbations and log it
            if self.log_plots:
                fig, _ , _, _ = plot.perturbations.local_perturbations_plot(perturbations)
                log_figure(kwargs['logs'], fig, f'{name}_perturbations', f'empirical_lipschitz{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
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

    def _get_distribution_labels_perturbations(self, **kwargs):
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
        distribution_labels = get_distribution_labels_perturbations(
            [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on],
        )
        auroc_mask = torch.ones_like(distribution_labels).bool()
        auroc_labels = torch.zeros_like(distribution_labels).bool()
        auroc_labels[distribution_labels == evaluation_constants.ID_CLASS_NO_OOD_CLASS_NBS] = True
        distribution_label_names = {
            evaluation_constants.OOD_CLASS_NO_ID_CLASS_NBS : f'Perturbed', 
            evaluation_constants.ID_CLASS_NO_OOD_CLASS_NBS : f'Unperturbed', 
        }
        return auroc_labels, auroc_mask, distribution_labels, distribution_label_names

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
        elif self.kind.lower() in ('perturbations', 'perturbation', 'noise'):
            return self._get_distribution_labels_perturbations(**kwargs)
        else:
            raise RuntimeError(f'Could not separate distribution labels (id vs ood) by unknown type {self.kind}.')

class OODDetection(OODSeparation):
    """ Pipeline member to perform OOD detection for a given metric. Evaluates AUROC scores and logs plots. """

    name = 'OODDetection'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def ood_detection(self, proxy, labels, proxy_name, auroc_labels, auroc_mask, distribution_labels, distribution_label_names,
                        plot_proxy_log_scale=True,**kwargs):
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
        """

        if plot_proxy_log_scale:
            proxy += 1e-10 # To be able to plot

        # Calculate area under the ROC for separating in-distribution (label 1) from out of distribution (label 0)
        roc_auc = roc_auc_score(auroc_labels[auroc_mask].cpu().long().numpy(), proxy[auroc_mask].cpu().numpy()) # higher proxy -> higher uncertainty
        kwargs['metrics'][f'auroc_{proxy_name}{self.suffix}'] = roc_auc
        log_metrics(kwargs['logs'], {f'auroc_{proxy_name}{self.suffix}' : roc_auc}, f'{proxy_name}_plots')

        # Calculate the area under the PR curve
        precision, recall, _ = precision_recall_curve(auroc_labels[auroc_mask].cpu().long().numpy(), proxy[auroc_mask].cpu().numpy())
        aucpr = auc(recall, precision)
        kwargs['metrics'][f'aucpr_{proxy_name}{self.suffix}'] = aucpr
        log_metrics(kwargs['logs'], {f'aucpr_{proxy_name}{self.suffix}' : aucpr}, f'{proxy_name}_plots')

        try:
            # Log histograms and metrics label-wise
            if self.log_plots:
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
                    }, f'{proxy_name}_statistics', step=label)
                fig, ax = plot_histograms(proxy.cpu(), y.cpu(), log_scale=plot_proxy_log_scale, kind='vertical', x_label=f'Proxy', y_label='Class')
                log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_all_classes{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Evaluated {proxy_name}.')
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not create label-wise plots for {proxy_name}.')

        try:
            if self.log_plots:
                fig, ax = plot_histograms(proxy.cpu(), distribution_labels.cpu(), 
                    label_names=distribution_label_names,
                    kind='vertical', kde=True, log_scale=plot_proxy_log_scale,  x_label=f'Proxy', y_label='Kind')
                log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_all_kinds{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Saved {proxy_name} (all kinds) histogram to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_histograms_all_kinds{self.suffix}.pdf')))
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not distribution label-wise plots for {proxy_name}.')

        try:
            if self.log_plots:
                fig, ax = plot_histograms(proxy[auroc_mask].cpu(), auroc_labels[auroc_mask].cpu().long(), 
                    label_names={0 : 'Out ouf distribution', 1 : 'In distribution'},
                    kind='overlapping', kde=True, log_scale=plot_proxy_log_scale,  x_label=f'Proxy', y_label='Kind')
                log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_id_vs_ood{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Saved {proxy_name} histogram (id vs ood) to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_histograms_id_vs_ood{self.suffix}.pdf')))
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not id vs ood plots for {proxy_name}.')

        # # Calculate average precision score
        # apr = average_precision_score(auroc_labels[auroc_mask].cpu().long().numpy(), proxy[auroc_mask].cpu().numpy()) # higher proxy -> higher uncertainty
        # kwargs['metrics'][f'apr_{proxy_name}{self.suffix}'] = apr
        # log_metrics(kwargs['logs'], {f'apr_{proxy_name}{self.suffix}' : apr}, f'{proxy_name}_plots')
        
class EvaluateLogitEnergy(OODDetection):
    """ Pipeline member to evaluate the Logit Energy curves of the model for in-distribution and out-of-distribution data. """

    name = 'EvaluateLogitEnergy'

    def __init__(self, gpus=0, evaluate_on=[data_constants.VAL], separate_distributions_by='train-label', 
                separate_distributions_tolerance=0.0, log_plots=True, temperature=1.0, **kwargs):
        super().__init__(separate_distributions_by=separate_distributions_by, 
                            separate_distributions_tolerance=separate_distributions_tolerance,
                            evaluate_on=evaluate_on,
                            **kwargs)
        self.gpus = gpus
        self.log_plots = log_plots
        self.temperature = temperature

    @property
    def configuration(self):
        return super().configuration | {
            'Log plots' : self.log_plots,
            'Temperature' : self.temperature,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        data_loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
        logits, labels = run_model_on_datasets(kwargs['model'], data_loaders, callbacks=[
                make_callback_get_logits(mask=True, ensemble_average=False),
                make_callback_get_ground_truth(mask=True),
            ], gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate)
        logits, labels = torch.cat(logits), torch.cat(labels) # Logits are of shape : N, n_classes, n_ensemble
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1) # N, n_ensemble
        energy = energy.mean(-1) # Average over ensemble members

        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)
        self.ood_detection(-energy, labels, 'logit-energy', auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
        
        return args, kwargs

class EvaluateSoftmaxEntropy(OODDetection):
    """ Pipeline member to evaluate the Softmax Entropy curves of the model for in-distribution and out-of-distribution data. """

    name = 'EvaluateSoftmaxEntropy'

    def __init__(self, gpus=0, evaluate_on=[data_constants.VAL], separate_distributions_by='train-label', 
                separate_distributions_tolerance=0.0, log_plots=True, **kwargs):
        super().__init__(separate_distributions_by=separate_distributions_by, 
                            separate_distributions_tolerance=separate_distributions_tolerance,
                            evaluate_on=evaluate_on,
                            **kwargs)
        self.gpus = gpus
        self.log_plots = log_plots

    @property
    def configuration(self):
        return super().configuration | {
            'Log plots' : self.log_plots,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        data_loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
        scores, labels = run_model_on_datasets(kwargs['model'], data_loaders, callbacks=[
                make_callback_get_predictions(mask=True, ensemble_average=False), # Average the prediction scores over the ensemble
                make_callback_get_ground_truth(mask=True),
            ], gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate)
        scores, labels = torch.cat(scores), torch.cat(labels) # Scores are of shape : N, n_classes, n_ensemble

        # Aleatoric uncertainty is the expected entropy
        expected_entropy = -(scores * torch.log2(scores + ENTROPY_EPS)).sum(1)
        expected_entropy = expected_entropy.mean(-1) # Get the expectation over all ensemble members

        # Epistemic uncertainty is the information gain, i.e. predictive uncertainty - aleatoric uncertainty
        avg_scores = scores.mean(-1) # Expectations of predictions in all ensemble members
        predictive_entropy = -(avg_scores * torch.log2(avg_scores + ENTROPY_EPS)).sum(1)

        max_scores, argmax_scores = scores.mean(-1).max(-1)

        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)
        self.ood_detection(-predictive_entropy, labels, 'total-predictive-entropy', auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
        self.ood_detection(max_scores, labels, 'max-score', auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
        if scores.size()[-1] > 1: # Some ensembling is used (ensembles, dropout, etc.), so epistemic and aleatoric estimates can be disentangled
            self.ood_detection(-expected_entropy, labels, 'expected-softmax-entropy', auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
            self.ood_detection(-(predictive_entropy - expected_entropy), labels, 'mutual-information', auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
            # Also use the empirical variance of the predicted class as proxy
            var = torch.var(scores, dim=-1) # Variance across ensemble, shape N x num_classes
            var_predicted_class = var[torch.arange(argmax_scores.size(0)), argmax_scores]
            self.ood_detection(1 / (var_predicted_class + VAR_EPS), labels, 'predicted-class-variance', auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)

        return args, kwargs

class FeatureDensity(OODDetection):
    """ Superclass for pipeline members that fit a feature density. """

    name = 'FeatureDensity'

    def __init__(self, gpus=0, fit_to=[data_constants.TRAIN], 
        fit_to_ground_truth_labels=[data_constants.TRAIN], evaluate_on=[data_constants.TEST], separate_distributions_by='train_label',
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

    @property
    def configuration(self):
        return super().configuration | {
            'Fit to' : self.fit_to,
            'Use ground truth labels for fit on' : self.fit_to_ground_truth_labels,
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
                    evaluate_on=[data_constants.VAL], density_types={}, dimensionality_reductions={}, gpus=0,
                    separate_distributions_by='train-label', separate_distributions_tolerance=0.0, seed=1337,
                    **kwargs):
        super().__init__(gpus=gpus, fit_to=fit_to, fit_to_ground_truth_labels=fit_to_ground_truth_labels, 
                        separate_distributions_by=separate_distributions_by,
                        separate_distributions_tolerance=separate_distributions_tolerance,
                        evaluate_on=evaluate_on, **kwargs)
        self.density_types = density_types
        self.dimensionality_reductions = dimensionality_reductions
        self.seed = seed

    @property
    def configuration(self):
        return super().configuration | {
            'Density types' : self.density_types,
            'Dimensionality Reductions' : self.dimensionality_reductions,
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
                dim_reduction.fit(features_to_fit)
                pipeline_log(f'{self.name} fitted dimensionality reduction {dim_reduction.compressed_name}')

                # TODO: dim-reductions transform per-class, but this is not supported here, so we just take any and set `per_class` to false in its constructor
                features_to_fit_reduced = torch.from_numpy(dim_reduction.transform(features_to_fit))
                features_to_evaluate_reduced = torch.from_numpy(dim_reduction.transform(features_to_evaluate))

                # Grid over feature space densities
                for density_type, density_grid in self.density_types.items():
                    keys_density = list(density_grid.keys())
                    for values_density in product(*[density_grid[k] for k in keys_density]):
                        density_config = {key : values_density[idx] for idx, key in enumerate(keys_density)}
                        density_model = get_density_model(
                            density_type=density_type, 
                            **density_config,
                            )
                        density_model.fit(features_to_fit_reduced, predictions_to_fit)
                        log_density = density_model(features_to_evaluate_reduced).cpu()
                        is_finite_density = torch.isfinite(log_density)

                        pipeline_log(f'{self.name} fitted density {density_model.compressed_name}')
                        proxy_name = f'{density_model.compressed_name}:{dim_reduction.compressed_name}'
                        self.ood_detection(log_density[is_finite_density], labels_to_evaluate[is_finite_density],
                            proxy_name,
                            auroc_labels[is_finite_density], auroc_mask[is_finite_density], distribution_labels[is_finite_density],
                            distribution_label_names, plot_proxy_log_scale=False, **kwargs
                        )
                        
                        if self.log_plots:
                            fig, ax = plot_density(features_to_fit, 
                                features_to_evaluate[is_finite_density], density_model, 
                                distribution_labels[is_finite_density], distribution_label_names, seed=self.seed)
                            log_figure(kwargs['logs'], fig, f'pca{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                            plt.close(fig)


        return args, kwargs

class VisualizeIDvsOOD(OODSeparation):
    """ Fits a 2d visualization to some feature space and separates id and ood data. """

    name = 'VisualizeIDvsOOD'

    def __init__(self, gpus=0, fit_to=[data_constants.TRAIN], layer=-2, dimensionality_reductions= ['pca'], **kwargs):
            super().__init__(**kwargs)
            self.gpus = gpus
            self.fit_to = fit_to
            self.layer = layer
            self.dimensionality_reductions = dimensionality_reductions

    @property
    def configuration(self):
        return super().configuration | {
            'Fit to' : self.fit_to,
            'Feature layer' : self.layer,
            'Dimensionality reductions' : self.dimensionality_reductions,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):


        features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], 
                                                                gpus=self.gpus, model_kwargs=self.model_kwargs_fit,
                                                                callbacks=[
                                                                    make_callback_get_features(layer=self.layer),
                                                                    make_callback_get_predictions(),
                                                                    make_callback_get_ground_truth(),
                                                                ])
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)
        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)

        for dimensionality_reduction in self.dimensionality_reductions:
            group = f'visualization_id_vs_ood_{dimensionality_reduction.lower()}{self.suffix}'
            dim_reduction = DimensionalityReduction(type = dimensionality_reduction, number_components=2)
            dim_reduction.fit(features)
            statistics = dim_reduction.statistics()
            if len(statistics) > 0:
                log_metrics(kwargs['logs'], statistics, group)
            transformed = dim_reduction.transform(features)
            
            fig, ax = plot_2d_features(torch.tensor(transformed), labels)
            log_figure(kwargs['logs'], fig, f'data_fit', group, kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Logged 2d{dimensionality_reduction} fitted to {self.fit_to}')
            plt.close(fig)

            features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], 
                                                                    gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate,
                                                                    callbacks=[
                                                                        make_callback_get_features(layer=self.layer),
                                                                        make_callback_get_predictions(),
                                                                        make_callback_get_ground_truth(),
                                                                    ])
            features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)
            transformed = dim_reduction.transform(features)
            fig, ax = plot_2d_features(torch.tensor(transformed), labels)
            log_figure(kwargs['logs'], fig, f'id_vs_ood_by_label', group, kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Logged 2d {dimensionality_reduction} fitted to {self.fit_to}, evaluated on {self.evaluate_on} by label')
            plt.close(fig)

            fig, ax = plot_2d_features(torch.tensor(transformed), distribution_labels, distribution_label_names)
            log_figure(kwargs['logs'], fig, f'id_vs_ood_by_distribution', group, kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Logged 2d {dimensionality_reduction} fitted to {self.fit_to}, evaluated on {self.evaluate_on} by in-distribution vs out-of-distribution')
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

        features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], gpus=self.gpus, model_kwargs=self.model_kwargs_fit,)
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)

        pca = PCA(n_components=self.num_components)
        projected = pca.fit_transform(features.cpu().numpy())

        log_embedding(kwargs['logs'], projected, f'pca_{self.suffix}', labels.cpu().numpy(), save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Fit feature space PCA with {self.num_components} components. Explained variance ratio {pca.explained_variance_ratio_}')

        if len(self.evaluate_on) > 0 and self.num_components > 2:
            warn(f'Attempting to evalute PCA on {self.evaluate_on} but dimension is {self.num_components} != 2. No plots created.')

        if self.num_components == 2:
            loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
            features, predictions, labels = run_model_on_datasets(kwargs['model'], loaders, gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate)
            for idx, data_name in enumerate(self.evaluate_on):

                projected = pca.fit_transform(features[idx].cpu().numpy())
                fig, ax = plot_2d_features(torch.tensor(projected), labels[idx])
                log_figure(kwargs['logs'], fig, f'pca_{self.suffix}_{data_name}_gnd', 'pca', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                plt.close(fig)

                fig, ax = plot_2d_features(torch.tensor(projected), predictions[idx].argmax(dim=1))
                log_figure(kwargs['logs'], fig, f'pca_{self.suffix}_{data_name}_predicted', 'pca', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
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

        features_all, predictions_all, labels_all = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate)
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
        results = run_model_on_datasets(kwargs['model'], [get_data_loader(data, kwargs['data_loaders']) for data in (self.data_before, self.data_after)], gpus=self.gpus, callbacks=callbacks, model_kwargs=self.model_kwargs)
        features, data, num_nbs_in_train_labels, num_nbs = results[0], results[1], results[2::2], results[3::2]

        idx_before, idx_after = vertex_intersection(data[0], data[1])
        shift = (features[0][idx_before] - features[1][idx_after]).norm(dim=1)
        
        # Log the feature shift by "in mask / not in mask"
        fig, ax = plot_histograms(shift.cpu() + FEATURE_SHIFT_EPS, data[0].mask[idx_before].cpu(), 
                label_names={True : f'In {self.data_before}', False : f'Not in {self.data_before}'}, log_scale=True, kind='overlapping', x_label='Feature Shift')
        log_figure(kwargs['logs'], fig, f'feature_shift_by_mask', f'inductive_feature_shift{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
        plt.close(fig)
        pipeline_log(f'Logged inductive feature shift for data {self.data_before} -> {self.data_after} by mask.')

        for k in range(1, receptive_field_size + 1):
            fraction = 1 - (num_nbs_in_train_labels[k - 1][1].float() / (num_nbs[k - 1][1] + 1e-12))
            fig, ax = plot_2d_histogram(shift.cpu() + FEATURE_SHIFT_EPS, fraction[idx_after], x_label='Log Feature Shift', y_label=f'Fraction of ood vertices in {k} neighbourhood', log_scale_x=True)
            log_figure(kwargs['logs'], fig, f'feature_shift_by_{k}_nbs', f'inductive_feature_shift{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
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
            make_callback_get_predictions(mask=False, ensemble_average=False), # Average the prediction scores over the ensemble
            make_callback_get_data(),
        ]
        for k in range(1, receptive_field_size + 1):
            callbacks += [
                make_callback_count_neighbours_with_labels(kwargs['config']['data']['train_labels'], k, mask=False),
                make_callback_count_neighbours(k, mask=False),
            ]
        results = run_model_on_datasets(kwargs['model'], [get_data_loader(data, kwargs['data_loaders']) for data in (self.data_before, self.data_after)], gpus=self.gpus, callbacks=callbacks, model_kwargs=self.model_kwargs)
        scores, data, num_nbs_in_train_labels, num_nbs = results[0], results[1], results[2::2], results[3::2]
        entropy = [
            -(score * torch.log2(score + ENTROPY_EPS)).sum(1).mean(-1) # Average over ensemble axis
            for score in scores # Expected entropy per dataset
        ]

        idx_before, idx_after = vertex_intersection(data[0], data[1]) 
        shift = -(entropy[0][idx_before] - entropy[1][idx_after])
        
        # Log the entropy shift by "in mask / not in mask"
        fig, ax = plot_histograms(shift.cpu() + 0, data[0].mask[idx_before].cpu(), 
                label_names={True : f'In {self.data_before}', False : f'Not in {self.data_before}'}, log_scale=False, kind='overlapping', x_label='Entropy Shift')
        log_figure(kwargs['logs'], fig, f'entropy_shift_by_mask', f'inductive_entropy_shift{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
        plt.close(fig)
        pipeline_log(f'Logged inductive entropy shift for data {self.data_before} -> {self.data_after} by mask.')

        for k in range(1, receptive_field_size + 1):
            fraction = 1 - (num_nbs_in_train_labels[k - 1][1].float() / (num_nbs[k - 1][1] + 1e-12))
            fig, ax = plot_2d_histogram(shift.cpu() + FEATURE_SHIFT_EPS, fraction[idx_after], x_label='Entropy Shift', y_label=f'Fraction of ood vertices in {k} neighbourhood', log_scale_x=False)
            log_figure(kwargs['logs'], fig, f'entropy_shift_by_{k}_nbs', f'inductive_entropy_shift{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
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

class PerturbData(PipelineMember):
    """ Pipeline member that creates a dataset with perturbed features. """

    name = 'PerturbData'

    def __init__(self, base_data=data_constants.OOD, dataset_name='unnamed-perturbation-dataset', 
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

class EvaluateAccuracy(OODSeparation):
    """ Pipeline member to evaluate the accuracy of the model on a dataset. Note: The dataset should follow the train labels. """

    name = 'EvaluateAccuracy'

    def __init__(self, evaluate_on=[data_constants.VAL_TRAIN_LABELS], evaluate_on_id_only=True, gpus=0, **kwargs):
        super().__init__(evaluate_on=evaluate_on, **kwargs)
        self.gpus = gpus

    @property
    def configuration(self):
        return super().configuration
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        
        predictions, labels, mask = run_model_on_datasets(
            kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], 
            gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate,
            callbacks = [
                evaluation.callbacks.make_callback_get_predictions(),
                evaluation.callbacks.make_callback_get_ground_truth(),
                evaluation.callbacks.make_callback_is_ground_truth_in_labels(kwargs['config']['data']['train_labels']),

            ])
        mask, predictions, labels = torch.cat(mask, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)
        is_id, is_id_mask, _, _ = self.get_distribution_labels(**kwargs)

        # Accuracy should only be computed for classes the model can actually predict
        predictions, labels, is_id_mask, is_id = predictions[mask], labels[mask], is_id_mask[mask], is_id[mask]

        _, hard = predictions.max(dim=-1)
        acc = (hard == labels).float().mean()
        acc_id = (hard == labels)[(is_id == 1) & is_id_mask].float().mean()
        acc_ood = (hard == labels)[(is_id == 0) & is_id_mask].float().mean()

        dataset_names = '-'.join(self.evaluate_on)
        kwargs['metrics'][f'accuracy_{dataset_names}{self.suffix}'] = acc.item()
        kwargs['metrics'][f'accuracy_id_{dataset_names}{self.suffix}'] = acc_id.item()
        kwargs['metrics'][f'accuracy_ood_{dataset_names}{self.suffix}'] = acc_ood.item()
        pipeline_log(f'Evaluated accuracy for {self.evaluate_on}.')

        return args, kwargs


class EvaluateCalibration(PipelineMember):
    """ Evaluates the calibration of a model. """

    name = 'EvaluateCalibration'

    def __init__(self, evaluate_on=[data_constants.VAL_TRAIN_LABELS], bins=10, gpus=0, **kwargs):
        super().__init__(**kwargs)
        self.evaluate_on = evaluate_on
        self.gpus = gpus
        self.bins = bins

    @property
    def configuration(self):
        return super().configuration | {
            'Evaluate on' : self.evaluate_on,
            'Bins' : self.bins,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        
        pred, gnd = run_model_on_datasets(kwargs['model'], [get_data_loader(dataset, kwargs['data_loaders']) for dataset in self.evaluate_on], gpus=self.gpus,
            callbacks = [
                make_callback_get_predictions(),
                make_callback_get_ground_truth(),
            ], model_kwargs=self.model_kwargs_evaluate)
        scores, gnd = torch.cat(pred, dim=0), torch.cat(gnd, dim=0)
        ece = expected_calibration_error(scores, gnd, bins=self.bins, eps=ECE_EPS)

        dataset_names = '-'.join(self.evaluate_on)
        kwargs['metrics'][f'ece_{dataset_names}{self.suffix}'] = ece
        log_metrics(kwargs['logs'], {f'ece_{dataset_names}{self.suffix}' : ece}, f'calibration')

        if self.log_plots:
            fig, ax = plot_calibration(scores, gnd, bins=self.bins, eps=ECE_EPS)
            log_figure(kwargs['logs'], fig, f'calibration_{dataset_names}{self.suffix}', f'calibration', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            plt.close(fig)
        
        pipeline_log(f'Logged calibration for {self.evaluate_on}')
        return args, kwargs


pipeline_members = [
    EvaluateEmpircalLowerLipschitzBounds,
    LogFeatures,
    FitFeatureSpacePCA,
    VisualizeIDvsOOD,
    PrintDatasetSummary,
    LogInductiveFeatureShift,
    LogInductiveSoftmaxEntropyShift,
    EvaluateLogitEnergy,
    EvaluateSoftmaxEntropy,
    FitFeatureDensityGrid,
    SubsetDataByLabel,
    PerturbData,
    EvaluateAccuracy,
    EvaluateCalibration,
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
                # Update settings that the member does not specify with the master configuration
                if 'log_plots' not in member:
                    member['log_plots'] = config['log_plots']
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
                pipeline_log(f'Running {member.print_name}...')
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
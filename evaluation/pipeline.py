import numpy as np
import torch
import evaluation.lipschitz
import plot.perturbations
import matplotlib.pyplot as plt
from evaluation.util import split_labels_into_id_and_ood, get_data_loader, run_model_on_datasets, count_neighbours_with_label, get_out_of_distribution_labels
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
from data.util import label_binarize, data_get_summary, labels_in_dataset, vertex_intersection
from warnings import warn
from itertools import product
from util import format_name, get_k_hop_neighbourhood
from copy import deepcopy
import traceback
import data.constants as data_constants

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

    def __str__(self):
        if self.perturbation_type.lower() == 'noise':
            stats = [
                f'\t Min perturbation : {self.min_perturbation}',
                f'\t Max perturbation : {self.max_perturbation}',
            ]
        elif self.perturbation_type.lower() == 'derangement':
            stats = [
                f'\t Permutation per Sample : {self.permute_per_sample}',
            ]


        return '\n'.join([
            self.print_name,
            f'\t Perturbation type : {self.perturbation_type}',
        ] + stats + [
            f'\t Number perturbations per sample : {self.num_perturbations_per_sample}',
            f'\t Number of perturbations : {self.num_perturbations}',
            f'\t Seed : {self.seed}',
            f'\t Evaluate on : {self.evaluate_on}',
        ])

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

class EvaluateSoftmaxEntropy(PipelineMember):
    """ Pipeline member to evaluate the Softmax Entropy curves of the model for in-distribution and out-of-distribution data. """

    name = 'EvaluateSoftmaxEntropy'

    def __init__(self, gpus=0, evaluate_on=[data_constants.VAL], **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.evaluate_on = evaluate_on

    def __str__(self):
        return '\n'.join([
            self.print_name,
            f'\t Evaluate on : {self.evaluate_on}',
        ])

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        data_loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
        entropy, labels = run_model_on_datasets(kwargs['model'], data_loaders, callbacks=[
                make_callback_get_softmax_entropy(mask=True),
                make_callback_get_ground_truth(mask=True),
            ], gpus=self.gpus)
        entropy, labels = torch.cat(entropy), torch.cat(labels)

        # Log histograms and metrics label-wise
        y = labels.cpu()
        for label in torch.unique(y):
            entropy_label = entropy[y == label]
            log_histogram(kwargs['logs'], entropy_label.cpu().numpy(), 'softmax_entropy', global_step=label, label_suffix=str(label.item()))
            log_metrics(kwargs['logs'], {
                f'{self.prefix}mean_softmax_entropy' : entropy_label.mean(),
                f'{self.prefix}std_softmax_entropy' : entropy_label.std(),
                f'{self.prefix}min_softmax_entropy' : entropy_label.min(),
                f'{self.prefix}max_softmax_entropy' : entropy_label.max(),
                f'{self.prefix}median_softmax_entropy' : entropy_label.median(),
            }, 'softmax_entropy', step=label)
        fig, ax = plot_histograms(entropy.cpu(), y.cpu(), log_scale=False, kind='vertical', x_label='Entropy', y_label='Class')
        log_figure(kwargs['logs'], fig, f'softmax_entropy_histograms_all_classes{self.suffix}', 'softmax_entropy_plots', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Evaluated softmax entropy for entire validation dataset (labels : {torch.unique(y).cpu().tolist()}).')
        plt.close(fig)

        mask_is_train_label, num_train_label_nbs, num_nbs = get_out_of_distribution_labels(
            [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on],
            len(kwargs['config']['model']['hidden_sizes']) + 1, # Receptive field of each vertex is num-hidden-layers + 1
            kwargs['config']['data']['train_labels'],
            mask = True,
        )
        mask_is_train_label = torch.cat(mask_is_train_label, dim=0) # N
        num_train_label_nbs = torch.stack([torch.cat(l, dim=0) for l in num_train_label_nbs]) # k, N
        num_nbs = torch.stack([torch.cat(l, dim=0) for l in num_nbs]) # k, N
        mask_has_non_train_label_nb = ((num_nbs - num_train_label_nbs) > 0).any(dim=0)
        
        # Plot the softmax-entropy distribution by train-label data i) with no non-train label neighbours ii) with at least one train label neighbour
        labels_to_plot = torch.empty_like(mask_is_train_label).long()
        labels_to_plot[~mask_is_train_label] = 0
        labels_to_plot[mask_is_train_label & (~mask_has_non_train_label_nb)] = 1
        labels_to_plot[mask_is_train_label & mask_has_non_train_label_nb] = 2
        
        k = len(kwargs['config']['model']['hidden_sizes']) + 1
        fig, ax = plot_histograms(entropy.cpu(), labels_to_plot.cpu(), 
            label_names={0 : 'ood', 1 : f'id, no ood in {k}-hops', 2 : f'id, ood in {k}-hops'},
            kind='overlapping', kde=True, log_scale=False,  x_label='Softmax-Entropy', y_label='Kind')
        log_figure(kwargs['logs'], fig, f'softmax_entropy_histograms_id_vs_ood{self.suffix}', 'softmax_entropy_plots', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Saved softmax-entropy histogram to ' + str(osp.join(kwargs['artifact_directory'], f'softmax_entropy_histograms_id_vs_ood{self.suffix}.pdf')))
        plt.close(fig)

        # Calculate area under the ROC for separating in-distribution (label 1) from out of distribution (label 0)
        roc_auc = roc_auc_score(~mask_is_train_label.long().numpy(), entropy.cpu().numpy()) # higher entropy -> higher uncertainty
        kwargs['metrics'][f'auroc_softmax_entropy{self.suffix}'] = roc_auc
        log_metrics(kwargs['logs'], {f'auroc_softmax_entropy{self.suffix}' : roc_auc}, 'softmax_entropy_plots')
        
        return args, kwargs

class PipelineFeatureDensity(PipelineMember):
    """ Superclass for pipeline members that fit a feature density. """

    def __init__(self, gpus=0, fit_to=[data_constants.TRAIN], 
        fit_to_ground_truth_labels=[data_constants.TRAIN], evaluate_on=[data_constants.TEST], **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.fit_to = fit_to
        self.fit_to_ground_truth_labels = fit_to_ground_truth_labels
        self.evaluate_on = evaluate_on

    def _get_features_and_labels_to_fit(self, **kwargs):
        features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], gpus=self.gpus)
        for idx, name in enumerate(self.fit_to):
            if name.lower() in self.fit_to_ground_truth_labels: # Override predictions with ground truth for training data
                predictions[idx] = label_binarize(labels[idx], num_classes=predictions[idx].size(1)).float()
        return torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)

    def _get_features_and_labels_to_evaluate(self, **kwargs):
        features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], gpus=self.gpus)
        return torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)

class FitFeatureDensityGrid(PipelineFeatureDensity):
    """ Pipeline member that fits a grid of several densities to the feature space of a model. """

    name = 'FitFeatureDensityGrid'

    def __init__(self, fit_to=[data_constants.TRAIN], fit_to_ground_truth_labels=[data_constants.TRAIN], 
                    evaluate_on=[data_constants.VAL], density_types={}, dimensionality_reductions={}, log_plots=False, gpus=0,
                    **kwargs):
        super().__init__(gpus=gpus, fit_to=fit_to, fit_to_ground_truth_labels=fit_to_ground_truth_labels, evaluate_on=evaluate_on, **kwargs)
        self.density_types = density_types
        self.dimensionality_reductions = dimensionality_reductions
        self.log_plots = log_plots

    def __str__(self):
        return '\n'.join([
            self.print_name,
            f'\t Fit to : {self.fit_to}',
            f'\t Ground truth labels for fitting used : {self.fit_to_ground_truth_labels}',
            f'\t Evaluate on : {self.evaluate_on}',
            f'\t Density Types : {list(self.density_types.keys())}',
            f'\t Dimensionality Reduction Types : {list(self.dimensionality_reductions.keys())}',
            f'\t Log plos : {self.log_plots}',
        ])
        
    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        # Only calculate data once
        features_to_fit, predictions_to_fit, labels_to_fit = self._get_features_and_labels_to_fit(**kwargs)
        # Note that for `self.fit_to_ground_truth_labels` data, the `predictions_to_fit` are overriden with a 1-hot ground truth
        features_to_evaluate, predictions_to_evaluate, labels_to_evaluate = self._get_features_and_labels_to_evaluate(**kwargs)
        mask_is_train_label, num_train_label_nbs, num_nbs = get_out_of_distribution_labels(
            [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on],
            len(kwargs['config']['model']['hidden_sizes']) + 1, # Receptive field of each vertex is num-hidden-layers + 1
            kwargs['config']['data']['train_labels'],
            mask = True,
        )
        mask_is_train_label = torch.cat(mask_is_train_label, dim=0) # N
        num_train_label_nbs = torch.stack([torch.cat(l, dim=0) for l in num_train_label_nbs]) # k, N
        num_nbs = torch.stack([torch.cat(l, dim=0) for l in num_nbs]) # k, N
        mask_has_non_train_label_nb = ((num_nbs - num_train_label_nbs) > 0).any(dim=0)

        # Plot the feature density distribution by train-label data i) with no non-train label neighbours ii) with at least one train label neighbour
        labels_to_plot = torch.empty_like(mask_is_train_label).long()
        labels_to_plot[~mask_is_train_label] = 0
        labels_to_plot[mask_is_train_label & (~mask_has_non_train_label_nb)] = 1
        labels_to_plot[mask_is_train_label & mask_has_non_train_label_nb] = 2

        aurocs = {}

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

                        suffix = f'{density_model.compressed_name}:{dim_reduction.compressed_name}'
                        metric_name = f'auroc:{suffix}{self.suffix}'
                        if metric_name in aurocs:
                            warn(f'Multiple configurations give auroc metric key {metric_name}')
                        else:
                            aurocs[metric_name] = roc_auc_score(mask_is_train_label.long().numpy(), log_density.cpu().numpy())
                            
                            if self.log_plots:
                                k = len(kwargs['config']['model']['hidden_sizes']) + 1
                                fig, ax = plot_histograms(log_density.cpu(), labels_to_plot.cpu(), 
                                    label_names={0 : 'ood', 1 : f'id, no ood in {k}-hops', 2 : f'id, ood in {k}-hops'},
                                    kind='overlapping', kde=True, log_scale=False,  x_label='Log-Density', y_label='Kind')
                                log_figure(kwargs['logs'], fig, f'feature_log_density_histograms{suffix}', 'feature_density_plots', save_artifact=kwargs['artifact_directory'])
                                pipeline_log(f'Saved feature density histogram to ' + str(osp.join(kwargs['artifact_directory'], f'feature_log_density_histograms{suffix}.pdf')))
                                plt.close(fig)

        log_metrics(kwargs['logs'], aurocs, 'feature_density_auroc')
        for k, auroc in aurocs.items():
            kwargs['metrics'][k] = auroc
        
        return args, kwargs

class FitFeatureDensity(PipelineFeatureDensity):
    """ Pipeline member that fits a density to the feature space of a model. """

    name = 'FitFeatureDensity'

    def __init__(self, density_type='unspecified', gpus=0, fit_to=[data_constants.TRAIN], fit_to_ground_truth_labels=[data_constants.TRAIN], 
                    evaluate_on=[data_constants.VAL], **kwargs):
        super().__init__(gpus=gpus, fit_to=fit_to, fit_to_ground_truth_labels=fit_to_ground_truth_labels, evaluate_on=evaluate_on, **kwargs)
        self.density = get_density_model(density_type=density_type, **kwargs)

    def __str__(self):
        return '\n'.join([
            self.print_name,
            f'\t Fit to : {self.fit_to}',
            f'\t Ground truth labels for fitting used : {self.fit_to_ground_truth_labels}',
            f'\t Evaluate on : {self.evaluate_on}',
            f'\t Density : {self.density}',
        ])

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        features, predictions, labels = self._get_features_and_labels_to_fit(**kwargs)
        self.density.fit(features, predictions)
        pipeline_log(f'Fitted density of type {self.density.name} to {self.fit_to}')

        features, predictions, labels = self._get_features_and_labels_to_evaluate(**kwargs)
        log_density = self.density(features.cpu()).cpu()
        
        # Log histograms and metrics label-wise
        y = labels.cpu()
        for label in torch.unique(y):
            log_density_label = log_density[y == label] + 1e-20
            log_histogram(kwargs['logs'], log_density_label.cpu().numpy(), 'feature_log_density', global_step=label, label_suffix=str(label.item()))
            log_metrics(kwargs['logs'], {
                f'{self.prefix}mean_feature_log_density' : log_density_label.mean(),
                f'{self.prefix}std_feature_log_density' : log_density_label.std(),
                f'{self.prefix}min_feature_log_density' : log_density_label.min(),
                f'{self.prefix}max_feature_log_density' : log_density_label.max(),
                f'{self.prefix}median_feature_log_density' : log_density_label.median(),
            }, 'feature_density', step=label)
        fig, ax = plot_histograms(log_density.cpu(), y.cpu(), log_scale=False, kind='vertical', x_label='Log-Density', y_label='Class')
        log_figure(kwargs['logs'], fig, f'feature_log_density_histograms_all_classes{self.suffix}', 'feature_class_density_plots', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Evaluated feature density for entire validation dataset (labels : {torch.unique(y).cpu().tolist()}).')
        plt.close(fig)

        mask_is_train_label, num_train_label_nbs, num_nbs = get_out_of_distribution_labels(
            [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on],
            len(kwargs['config']['model']['hidden_sizes']) + 1, # Receptive field of each vertex is num-hidden-layers + 1
            kwargs['config']['data']['train_labels'],
            mask = True,
        )
        mask_is_train_label = torch.cat(mask_is_train_label, dim=0) # N
        num_train_label_nbs = torch.stack([torch.cat(l, dim=0) for l in num_train_label_nbs]) # k, N
        num_nbs = torch.stack([torch.cat(l, dim=0) for l in num_nbs]) # k, N
        mask_has_non_train_label_nb = ((num_nbs - num_train_label_nbs) > 0).any(dim=0)
        
        # Plot the feature density distribution by train-label data i) with no non-train label neighbours ii) with at least one train label neighbour
        labels_to_plot = torch.empty_like(mask_is_train_label).long()
        labels_to_plot[~mask_is_train_label] = 0
        labels_to_plot[mask_is_train_label & (~mask_has_non_train_label_nb)] = 1
        labels_to_plot[mask_is_train_label & mask_has_non_train_label_nb] = 2
        
        k = len(kwargs['config']['model']['hidden_sizes']) + 1
        fig, ax = plot_histograms(log_density.cpu(), labels_to_plot.cpu(), 
            label_names={0 : 'ood', 1 : f'id, no ood in {k}-hops', 2 : f'id, ood in {k}-hops'},
            kind='overlapping', kde=True, log_scale=False,  x_label='Log-Density', y_label='Kind')
        log_figure(kwargs['logs'], fig, f'feature_log_density_histograms_id_vs_ood{self.suffix}', 'feature_density_plots', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Saved feature density histogram to ' + str(osp.join(kwargs['artifact_directory'], f'feature_log_density_histograms_id_vs_ood_{self.suffix}.pdf')))
        plt.close(fig)

        # Calculate area under the ROC for separating in-distribution (label 1) from out of distribution (label 0)
        roc_auc = roc_auc_score(mask_is_train_label.long().numpy(), log_density.cpu().numpy())
        kwargs['metrics'][f'auroc{self.suffix}'] = roc_auc
        log_metrics(kwargs['logs'], {f'auroc{self.suffix}' : roc_auc}, 'feature_density_plots')
        
        return args, kwargs

class FitFeatureSpacePCAIDvsOOD(PipelineMember):
    """ Fits a 2d PCA to the feature space and separates id and ood data. """

    name = 'FitFeatureSpacePCAIDvsOOD'

    def __init__(self, gpus=0, fit_to=['train'], evaluate_on=['val'], **kwargs):
            super().__init__(**kwargs)
            self.gpus = gpus
            self.fit_to = fit_to
            self.evaluate_on = evaluate_on

    def __str__(self):
        return '\n'.join([
            self.print_name,
            f'\t Fit to : {self.fit_to}',
            f'\t Evaluate on : {self.evaluate_on}',
        ])

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

        labels_id_ood = split_labels_into_id_and_ood(labels.cpu(), set(kwargs['config']['data']['train_labels']), id_label=_ID_LABEL, ood_label=_OOD_LABEL)
        fig, ax = plot_2d_features(torch.tensor(transformed), labels_id_ood)
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
    def __str__(self):
        return '\n'.join([
            self.print_name,
            f'\t Fit to : {self.fit_to}',
            f'\t Evaluate on : {self.evaluate_on}',
            f'\t Number of components : {self.num_components}',
        ])

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
                # Get the label-to-name mapping
                for data in loaders[idx]:
                    idx_to_label = {idx : label for label, idx in data.label_to_idx.items()}
                    break
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

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Evaluate on : {self.evaluate_on}',
        ])
    
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

    def __str__(self):
        return '\n'.join([
            self.print_name,
            f'\t Evaluate on : {self.evaluate_on}',
        ])

    def __call__(self, *args, **kwargs):
        for name in self.evaluate_on:
            loader = get_data_loader(name, kwargs['data_loaders'])
            print(f'# Data summary : {name}')
            print(data_get_summary(loader.dataset, prefix='\t'))

        return args, kwargs

class LogInductiveFeatureShift(PipelineMember):
    """ Logs the feature shift in vertices of the train-graph when re-introducing new edges of the val-graph. """

    name = 'LogInductiveFeatureShift'

    def __init__(self, gpus=0, data_before='train', data_after='val', **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.data_before = data_before
        self.data_after = data_after

    def __str__(self):
        return '\n'.join([
            self.print_name,
            f'\t Evaluate before : {self.data_before}',
            f'\t Evaluate train : {self.data_after}',
        ])
    
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

    def __init__(self, gpus=0, data_before='train', data_after='val', **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.data_before = data_before
        self.data_after = data_after

    def __str__(self):
        return '\n'.join([
            self.print_name,
            f'\t Evaluate before : {self.data_before}',
            f'\t Evaluate train : {self.data_after}',
        ])
    
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

pipeline_members = [
    EvaluateEmpircalLowerLipschitzBounds,
    FitFeatureDensity, # Deprecated: Use the Grid instead
    # EvaluateFeatureDensity,  # Merged into : FitFeatureDensity (why would you want to fit and not evaluate anyway??)
    LogFeatures,
    FitFeatureSpacePCA,
    FitFeatureSpacePCAIDvsOOD,
    PrintDatasetSummary,
    LogInductiveFeatureShift,
    LogInductiveSoftmaxEntropyShift,
    EvaluateSoftmaxEntropy,
    FitFeatureDensityGrid,
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
import numpy as np
import torch
import evaluation.lipschitz
import plot.perturbations
from evaluation.util import split_labels_into_id_and_ood, get_data_loader, feature_extraction
from plot.density import plot_2d_log_density, plot_log_density_histograms
from plot.features import plot_2d_features
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from model.density import get_density_model
from evaluation.logging import log_figure, log_histogram, log_embedding, log_metrics
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import os.path as osp
from data.util import label_binarize, data_get_summary
from warnings import warn

_ID_LABEL, _OOD_LABEL = 1, 0

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
            self.name,
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
        
        return args, kwargs

class FitFeatureDensity(PipelineMember):
    """ Pipeline member that fits a density to the feature space of a model. """

    name = 'FitFeatureDensity'

    def __init__(self, density_type='unspecified', gpus=0, fit_to=['train'], fit_to_ground_truth_labels=[], **kwargs):
        super().__init__(**kwargs)
        self.density = get_density_model(density_type=density_type, **kwargs)
        self.gpus = gpus
        self.fit_to = fit_to
        self.fit_to_ground_truth_labels = fit_to_ground_truth_labels

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Fit to : {self.fit_to}',
            f'\t Ground truth labels for fitting used : {self.fit_to_ground_truth_labels}',
            f'\t Density : {self.density}',
        ])

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        features, predictions, labels = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], gpus=self.gpus)
        for idx, name in enumerate(self.fit_to):
            if name.lower() in self.fit_to_ground_truth_labels:
                # Override predictions with ground truth for training data
                predictions[idx] = label_binarize(labels[idx], num_classes=predictions[idx].size(1)).float()
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)

        self.density.fit(features, predictions)
        if 'feature_density' in kwargs:
            pipeline_log('Density was already fit to features, overwriting...')
        kwargs['feature_density'] = self.density
        pipeline_log(f'Fitted density of type {self.density.name} to {self.fit_to}')
        return args, kwargs

class EvaluateFeatureDensity(PipelineMember):
    """ 
    Pipeline member that evaluates the feature density at each sample in the validation set.
    It also logs histograms and statistics.
    """

    name = 'EvaluateFeatureDensity'

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

        features, predictions, labels = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], gpus=self.gpus)
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)

        log_density = kwargs['feature_density'](features.cpu()).cpu()
        
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
        fig, ax = plot_log_density_histograms(log_density.cpu(), y.cpu(), overlapping=False)
        log_figure(kwargs['logs'], fig, f'feature_log_density_histograms_all_classes{self.suffix}', 'feature_class_density_plots', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Evaluated feature density for entire validation dataset (labels : {torch.unique(y).cpu().tolist()}).')

        # Split into in-distribution and out-of-distribution
        labels_id_ood = split_labels_into_id_and_ood(y.cpu(), set(kwargs['config']['data']['train_labels']), id_label=_ID_LABEL, ood_label=_OOD_LABEL)
        fig, ax = plot_log_density_histograms(log_density.cpu(), labels_id_ood.cpu(), label_names={_ID_LABEL : 'id', _OOD_LABEL : 'ood'})
        log_figure(kwargs['logs'], fig, f'feature_log_density_histograms_id_vs_ood{self.suffix}', 'feature_density_plots', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Saved feature density histogram to ' + str(osp.join(kwargs['artifact_directory'], f'feature_log_density_histograms_id_vs_ood_{self.suffix}.pdf')))

        # Calculate area under the ROC for separating in-distribution (label 1) from out of distribution (label 0)
        roc_auc = roc_auc_score(labels_id_ood.numpy(), log_density.cpu().numpy())
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
            self.name,
            f'\t Fit to : {self.fit_to}',
            f'\t Evaluate on : {self.evaluate_on}',
        ])

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        features, predictions, labels = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], gpus=self.gpus)
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(features.cpu().numpy())
        fig, ax = plot_2d_features(torch.tensor(transformed), labels)
        log_figure(kwargs['logs'], fig, f'pca_2d_id_vs_ood{self.suffix}_visualization_data_fit', 'feature_space_pca_id_vs_ood', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Logged 2d pca fitted to {self.fit_to}')

        features, predictions, labels = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], gpus=self.gpus)
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)
        transformed = pca.transform(features.cpu().numpy())
        fig, ax = plot_2d_features(torch.tensor(transformed), labels)
        log_figure(kwargs['logs'], fig, f'pca_2d_id_vs_ood{self.suffix}_visualization_by_label', 'feature_space_pca_id_vs_ood', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Logged 2d pca fitted to {self.fit_to}, evaluated on {self.evaluate_on} by label')

        labels_id_ood = split_labels_into_id_and_ood(labels.cpu(), set(kwargs['config']['data']['train_labels']), id_label=_ID_LABEL, ood_label=_OOD_LABEL)
        fig, ax = plot_2d_features(torch.tensor(transformed), labels_id_ood)
        log_figure(kwargs['logs'], fig, f'pca_2d_id_vs_ood{self.suffix}_visualization_by_id_vs_ood', 'feature_space_pca_id_vs_ood', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Logged 2d pca fitted to {self.fit_to}, evaluated on {self.evaluate_on} by in-distribution vs out-of-distribution')

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
            self.name,
            f'\t Fit to : {self.fit_to}',
            f'\t Evaluate on : {self.evaluate_on}',
            f'\t Number of components : {self.num_components}',
        ])

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        features, predictions, labels = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], gpus=self.gpus)
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)

        pca = PCA(n_components=self.num_components)
        projected = pca.fit_transform(features.cpu().numpy())

        log_embedding(kwargs['logs'], projected, f'pca_{self.suffix}', labels.cpu().numpy(), save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Fit feature space PCA with {self.num_components} components. Explained variance ratio {pca.explained_variance_ratio_}')

        if len(self.evaluate_on) > 0 and self.num_components > 2:
            warn(f'Attempting to evalute PCA on {self.evaluate_on} but dimension is {self.num_components} != 2. No plots created.')

        if self.num_components == 2:
            loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
            features, predictions, labels = feature_extraction(kwargs['model'], loaders, gpus=self.gpus)
            for idx, data_name in enumerate(self.evaluate_on):
                # Get the label-to-name mapping
                for data in loaders[idx]:
                    idx_to_label = {idx : label for label, idx in data.label_to_idx.items()}
                    break
                projected = pca.fit_transform(features[idx].cpu().numpy())
                fig, ax = plot_2d_features(torch.tensor(projected), labels[idx])
                log_figure(kwargs['logs'], fig, f'pca_{self.suffix}_{data_name}_gnd', 'pca', save_artifact=kwargs['artifact_directory'])
                fig, ax = plot_2d_features(torch.tensor(projected), predictions[idx].argmax(dim=1))
                log_figure(kwargs['logs'], fig, f'pca_{self.suffix}_{data_name}_predicted', 'pca', save_artifact=kwargs['artifact_directory'])

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

        features_all, predictions_all, labels_all = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], gpus=self.gpus)
        for name, predictions, features, labels in zip(self.evaluate_on, features_all, predictions_all, labels_all):
            log_embedding(kwargs['logs'], features.cpu().numpy(), f'{name}_features', labels.cpu().numpy(), save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Logged features (size {features.size()}) of dataset {name}')
            
        return args, kwargs

class PrintDatasetSummary(PipelineMember):
    """ Pipeline member that prints dataset statistics. """

    name = 'PrintDatasetSummary'

    def __init__(self, gpus=0, evaluate_on=['train', 'val'], **kwargs):
        self.evaluate_on = evaluate_on

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Evaluate on : {self.evaluate_on}',
        ])

    def __call__(self, *args, **kwargs):
        for name in self.evaluate_on:
            loader = get_data_loader(name, kwargs['data_loaders'])
            print(f'# Data summary : {name}')
            print(data_get_summary(loader.dataset, prefix='\t'))

        return args, kwargs

pipeline_members = [
    EvaluateEmpircalLowerLipschitzBounds,
    FitFeatureDensity,
    EvaluateFeatureDensity,
    LogFeatures,
    FitFeatureSpacePCA,
    FitFeatureSpacePCAIDvsOOD,
    PrintDatasetSummary,
]

class Pipeline:
    """ Pipeline for stuff to do after a model has been trained. """

    def __init__(self, members: list, config: dict, gpus=0, ignore_exceptions=False):
        self.members = []
        self.ignore_exceptions = ignore_exceptions
        for idx, member in enumerate(members):
            for _class in pipeline_members:
                if member['type'].lower() == _class.name.lower():
                    self.members.append(_class(
                        gpus=gpus,
                        pipeline_idx = idx,
                        **member
                    ))
                    break
            else:
                raise RuntimeError(f'Unrecognized evaluation pipeline member {member}')

    def __call__(self, *args, **kwargs):
        for member in self.members:
            try:
                args, kwargs = member(*args, **kwargs)
            except Exception as e:
                pipeline_log(f'{member.name} FAILED. Reason: "{e}"')
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
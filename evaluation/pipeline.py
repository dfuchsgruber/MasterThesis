import numpy as np
import torch
import evaluation.lipschitz
import plot.perturbations
from evaluation.util import split_labels_into_id_and_ood, get_data_loader, feature_extraction
from plot.density import plot_2d_log_density, plot_log_density_histograms
from plot.features import plot_2d_features
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from model.density import FeatureSpaceDensityGaussianPerClass
from evaluation.logging import log_figure, log_histogram, log_embedding, log_metrics
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
import os.path as osp

_ID_LABEL, _OOD_LABEL = 0, 1

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
            seed=None, gpus=0, evaluate_on=['val'], **kwargs):
        super().__init__(**kwargs)
        self.num_perturbations_per_sample = num_perturbations_per_sample
        self.min_perturbation = min_perturbation
        self.max_perturbation = max_perturbation
        self.num_perturbations = num_perturbations
        self.seed = None
        self.gpus = gpus
        self.evaluate_on = evaluate_on

    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Number perturbations per sample : {self.num_perturbations_per_sample}',
            f'\t Min perturbation : {self.min_perturbation}',
            f'\t Max perturbation : {self.max_perturbation}',
            f'\t Number of perturbations : {self.num_perturbations}',
            f'\t Seed : {self.seed}',
            f'\t Evaluate on : {self.evaluate_on}',
        ])

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        for name in self.evaluate_on:
            loader = get_data_loader(name, kwargs)
            assert len(loader) == 1, f'Empirical local lipschitz evaluation is currently only supported for semi-supervised tasks.'
            for dataset in loader:
                break # Just want the single dataset
            if self.gpus > 0:
                dataset = dataset.to('cuda')
                kwargs['model'] = kwargs['model'].to('cuda')

            perturbations = evaluation.lipschitz.local_perturbations(kwargs['model'], dataset,
                perturbations=np.linspace(self.min_perturbation, self.max_perturbation, self.num_perturbations),
                num_perturbations_per_sample=self.num_perturbations_per_sample, seed=self.seed)
            pipeline_log(f'Created {self.num_perturbations_per_sample} perturbations in linspace({self.min_perturbation:.2f}, {self.max_perturbation:.2f}, {self.num_perturbations}) for validation samples.')
            smean, smedian, smax, smin = evaluation.lipschitz.local_lipschitz_bounds(perturbations)
            log_metrics(kwargs['logs'], {
                f'{name}_slope_mean_perturbation' : smean,
                f'{name}_slope_median_perturbation' : smedian,
                f'{name}_slope_max_perturbation' : smax,
                f'{name}_slope_min_perturbation' : smin,
            }, 'empirical_lipschitz')
            # Plot the perturbations and log it
            fig, _ , _, _ = plot.perturbations.local_perturbations_plot(perturbations)
            log_figure(kwargs['logs'], fig, f'{name}_perturbations', 'empirical_lipschitz', save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Logged input vs. output perturbation plot for dataset {name}.')
        
        return args, kwargs

class FitFeatureDensity(PipelineMember):
    """ Pipeline member that fits a density to the feature space of a model. """

    name = 'FitFeatureDensity'

    def __init__(self, density_type=FeatureSpaceDensityGaussianPerClass.name, gpus=0, fit_to=['train'], **kwargs):
        super().__init__(**kwargs)
        self.density_type = density_type.lower()
        if self.density_type == FeatureSpaceDensityGaussianPerClass.name.lower():
            self.density = FeatureSpaceDensityGaussianPerClass(**kwargs)
            self.fit_to = fit_to
        else:
            raise RuntimeError(f'Unsupported feature space density {self.density_type}')
        self.gpus = gpus

    
    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t ID : {self.id}'
            f'\t Fit to : {self.fit_to}',
            f'\t Density : {self.density}',
        ])

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        features, labels = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs) for name in self.fit_to], gpus=self.gpus)
        features, labels = torch.cat(features, dim=0), torch.cat(labels)

        self.density.fit(features, labels)
        if 'feature_density' in kwargs:
            pipeline_log('Density was already fit to features, overwriting...')
        kwargs['feature_density'] = self.density
        pipeline_log(f'Fitted density of type {self.density_type} to training data features.')
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

        features, labels = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs) for name in self.evaluate_on], gpus=self.gpus)
        features, labels = torch.cat(features, dim=0), torch.cat(labels)

        density = kwargs['feature_density'](features.cpu()).cpu()
        
        # Log histograms and metrics label-wise
        y = labels.cpu()
        for label in torch.unique(y):
            log_density_label = torch.log(density[y == label] + 1e-20)
            log_histogram(kwargs['logs'], log_density_label.cpu().numpy(), 'feature_log_density', global_step=label, label_suffix=str(label.item()))
            log_metrics(kwargs['logs'], {
                f'{self.prefix}mean_feature_log_density' : log_density_label.mean(),
                f'{self.prefix}std_feature_log_density' : log_density_label.std(),
                f'{self.prefix}min_feature_log_density' : log_density_label.min(),
                f'{self.prefix}max_feature_log_density' : log_density_label.max(),
                f'{self.prefix}median_feature_log_density' : log_density_label.median(),
            }, 'feature_density', step=label)
        fig, ax = plot_log_density_histograms(torch.log(density.cpu() + 1e-20), y.cpu(), overlapping=False)
        log_figure(kwargs['logs'], fig, f'feature_log_density_histograms_all_classes{self.suffix}', 'feature_density_plots', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Evaluated feature density for entire validation dataset (labels : {torch.unique(y).cpu().tolist()}).')

        # Split into in-distribution and out-of-distribution
        labels_id_ood = split_labels_into_id_and_ood(y.cpu(), set(kwargs['config']['data']['train_labels']), id_label=_ID_LABEL, ood_label=_OOD_LABEL)
        fig, ax = plot_log_density_histograms(torch.log(density.cpu()), labels_id_ood.cpu(), label_names={_ID_LABEL : 'id', _OOD_LABEL : 'ood'})
        log_figure(kwargs['logs'], fig, f'feature_log_density_histograms_id_vs_ood{self.suffix}', 'feature_density_plots', save_artifact=kwargs['artifact_directory'])

        pipeline_log(f'Saved feature density histogram to ' + str(osp.join(kwargs['artifact_directory'], f'feature_log_density_histograms_id_vs_ood_{self.suffix}.pdf')))

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

        features, labels = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs) for name in self.fit_to], gpus=self.gpus)
        features, labels = torch.cat(features, dim=0), torch.cat(labels)
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(features.cpu().numpy())
        fig, ax = plot_2d_features(torch.tensor(transformed), labels)
        log_figure(kwargs['logs'], fig, f'pca_2d_id_vs_ood{self.suffix}_visualization_data_fit', 'feature_space_pca_id_vs_ood', save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Logged 2d pca fitted to {self.fit_to}')

        features, labels = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs) for name in self.evaluate_on], gpus=self.gpus)
        features, labels = torch.cat(features, dim=0), torch.cat(labels)
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

    def __init__(self, gpus=0, fit_to=['train'], num_components=16, **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.fit_to = fit_to
        self.num_components = num_components
    def __str__(self):
        return '\n'.join([
            self.name,
            f'\t Fit to : {self.fit_to}',
            f'\t Number of components : {self.num_components}',
        ])

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        features, labels = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs) for name in self.fit_to], gpus=self.gpus)
        features, labels = torch.cat(features, dim=0), torch.cat(labels)

        pca = PCA(n_components=self.num_components)
        projected = pca.fit_transform(features.cpu().numpy())

        if self.num_components == 2:
            fig, ax = plot_2d_features(torch.tensor(projected), labels)
            log_figure(kwargs['logs'], fig, f'pca_{self.suffix}_visualization', 'pca', save_artifact=kwargs['artifact_directory'])

        log_embedding(kwargs['logs'], projected, f'pca_{self.suffix}', labels.cpu().numpy(), save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Fit feature space PCA with {self.num_components} components. Explained variance ratio {pca.explained_variance_ratio_}')
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

        features_all, labels_all = feature_extraction(kwargs['model'], [get_data_loader(name, kwargs) for name in self.evaluate_on], gpus=self.gpus)
        for name, features, labels in zip(self.evaluate_on, features_all, labels_all):
            log_embedding(kwargs['logs'], features.cpu().numpy(), f'{name}_features', labels.cpu().numpy(), save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Logged features (size {features.size()}) of dataset {name}')
            
        return args, kwargs

pipeline_members = [
    EvaluateEmpircalLowerLipschitzBounds,
    FitFeatureDensity,
    EvaluateFeatureDensity,
    LogFeatures,
    FitFeatureSpacePCA,
    FitFeatureSpacePCAIDvsOOD,
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
                        _idx = idx,
                        **member
                    ))
                    break
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

    def __str__(self):
        return '\n'.join([
            'Evaluation Pipeline : ',
        ] + [
            f'\t{member}' for member in self.members
        ])


def pipeline_log(string):
    print(f'EVALUATION PIPELINE - {string}')
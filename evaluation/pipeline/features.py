import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .base import *
import data.constants as dconstants
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *
from plot.features import plot_2d_features

@register_pipeline_member
class FitFeatureSpacePCA(PipelineMember):
    """ Fits PCA to the feature space using training and validation data. """

    name = 'FitFeatureSpacePCA'

    def __init__(self, gpus=0, fit_to=[dconstants.TRAIN], evaluate_on=[dconstants.VAL], num_components=16, **kwargs):
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

        log_embedding(kwargs['logs'], projected, f'pca_{self.suffix}', kwargs['artifacts'], labels=labels.cpu().numpy(), save_artifact=kwargs['artifact_directory'])
        pipeline_log(f'Fit feature space PCA with {self.num_components} components. Explained variance ratio {pca.explained_variance_ratio_}')

        if len(self.evaluate_on) > 0 and self.num_components > 2:
            warn(f'Attempting to evalute PCA on {self.evaluate_on} but dimension is {self.num_components} != 2. No plots created.')

        if self.num_components == 2:
            loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
            features, predictions, labels = run_model_on_datasets(kwargs['model'], loaders, gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate)
            for idx, data_name in enumerate(self.evaluate_on):

                projected = pca.transform(features[idx].cpu().numpy())
                fig, ax = plot_2d_features(torch.tensor(projected), labels[idx])
                log_figure(kwargs['logs'], fig, f'pca_{self.suffix}_{data_name}_gnd', 'pca', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                plt.close(fig)

                fig, ax = plot_2d_features(torch.tensor(projected), predictions[idx].argmax(dim=1))
                log_figure(kwargs['logs'], fig, f'pca_{self.suffix}_{data_name}_predicted', 'pca', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                plt.close(fig)

        kwargs['feature_space_pca'] = pca

        return args, kwargs

@register_pipeline_member
class LogFeatures(PipelineMember):
    """ Pipeline member that logs the features of the validation data. """

    name = 'LogFeatures'

    def __init__(self, gpus=0, evaluate_on=[dconstants.VAL], **kwargs):
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

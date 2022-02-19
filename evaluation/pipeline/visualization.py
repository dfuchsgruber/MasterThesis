import torch
import matplotlib.pyplot as plt

from .base import *
import data.constants as dconstants
from .uncertainty_quantification import OODSeparation
import evaluation.callbacks
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *
from model.dimensionality_reduction import DimensionalityReduction
from plot.features import plot_2d_features

@register_pipeline_member
class VisualizeIDvsOOD(OODSeparation):
    """ Fits a 2d visualization to some feature space and separates id and ood data. """

    name = 'VisualizeIDvsOOD'

    def __init__(self, gpus=0, fit_to=[dconstants.TRAIN], layer=-2, dimensionality_reductions= ['pca'], **kwargs):
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
                                                                    evaluation.callbacks.make_callback_get_features(layer=self.layer),
                                                                    evaluation.callbacks.make_callback_get_predictions(),
                                                                    evaluation.callbacks.make_callback_get_ground_truth(),
                                                                ])
        features, predictions, labels = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)
        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_ood_distribution_labels(**kwargs)

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
                                                                        evaluation.callbacks.make_callback_get_features(layer=self.layer),
                                                                        evaluation.callbacks.make_callback_get_predictions(),
                                                                        evaluation.callbacks.make_callback_get_ground_truth(),
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

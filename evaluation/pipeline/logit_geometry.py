import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
import matplotlib.pyplot as plt

from .base import *
from .ood import OODDetection, OODSeparation
import data.constants as dconstants
import evaluation.callbacks
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *
import plot.logit_geometry
import util

@register_pipeline_member
class EvaluateLogitGeometry(OODDetection):
    """ Pipeline member to evaluate the geometry (norm and angle) of logit space. """

    name = 'EvaluateLogitGeometry'

    def __init__(self, gpus=0, **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus

    @property
    def configuration(self):
        return super().configuration

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        cfg: configuration.ExperimentConfiguration = kwargs['config']
        features, labels, idx_to_label = run_model_on_datasets(
            kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], 
            gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate,
            callbacks = [
                evaluation.callbacks.make_callback_get_features(),
                evaluation.callbacks.make_callback_get_ground_truth(),
                evaluation.callbacks.make_callback_get_attribute(lambda data, output: {idx.item() : label for label, idx in data.label_to_idx.items()})
            ])
        features, labels = torch.cat(features, dim=0), torch.cat(labels)
        idx_to_label = reduce(lambda a, b: {**a, **b}, idx_to_label)
        is_id, id_ood_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)

        w = kwargs['model'].get_output_weights().detach().cpu().numpy()
        x = features.numpy()
        cos = cosine_similarity(x, w)
        x_norm = np.linalg.norm(x, ord=2, axis=-1)

        # Use the max cosine similarity as a proxy
        self.ood_detection(
            torch.tensor(cos.max(1)),
            labels,
            'logit_cosine_similarity',
            is_id,
            id_ood_mask,
            distribution_labels,
            distribution_label_names,
            plot_proxy_log_scale=False,
            **kwargs,
        )
        # Use the feature norm as a proxy (confidence)
        self.ood_detection(
            torch.tensor(x_norm),
            labels,
            'feature_norm',
            is_id,
            id_ood_mask,
            distribution_labels,
            distribution_label_names,
            plot_proxy_log_scale=False,
            **kwargs,
        )

        if self.log_plots:

            # Plot by ground truth class label
            fig, ax = plot.logit_geometry.plot_norms(
                x_norm, labels.cpu().numpy(), idx_to_label, norm_label='Feature Norm'
            )
            log_figure(kwargs['logs'], fig, f'feature_norm_by_class{self.suffix}', f'logit_geometry{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Saved feature norm by class' + str(osp.join(kwargs['artifact_directory'], f'feature_norm_by_class{self.suffix}.pdf')))
            plt.close(fig)

            fig, ax = plot.logit_geometry.plot_logit_cosine_angles(
                cos, labels.cpu().numpy(), idx_to_label, figsize=(15, 15),
            )
            log_figure(kwargs['logs'], fig, f'cosine_by_class{self.suffix}', f'logit_geometry{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Saved cosine angle by class' + str(osp.join(kwargs['artifact_directory'], f'cosine_by_class{self.suffix}.pdf')))
            plt.close(fig)

            # Plot by id / ood
            fig, ax = plot.logit_geometry.plot_norms(
                x_norm[id_ood_mask], is_id[id_ood_mask].cpu().numpy(), {True : 'In distribution', False : 'Out of Distribution'}, norm_label='Feature Norm', plot_labels=True,
            )
            log_figure(kwargs['logs'], fig, f'feature_norm_by_id_vs_ood{self.suffix}', f'logit_geometry{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Saved feature norm by id / ood' + str(osp.join(kwargs['artifact_directory'], f'feature_norm_by_id_vs_ood{self.suffix}.pdf')))
            plt.close(fig)
    
            fig, ax = plot.logit_geometry.plot_logit_cosine_angles(
                cos[id_ood_mask], is_id[id_ood_mask].cpu().numpy(), {True : 'In distribution', False : 'Out of Distribution'}, class_labels=idx_to_label, figsize=(15, 15), plot_labels=True,
            )
            log_figure(kwargs['logs'], fig, f'cosine_by_id_vs_ood{self.suffix}', f'logit_geometry{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            pipeline_log(f'Saved cosine angle by id / ood' + str(osp.join(kwargs['artifact_directory'], f'cosine_by_is_id_vs_ood{self.suffix}.pdf')))
            plt.close(fig)

        return args, kwargs
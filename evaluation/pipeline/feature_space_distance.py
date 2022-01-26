from evaluation.callbacks import make_callback_get_features, make_callback_get_ground_truth
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl


from .base import *
import data.constants as dconstants
from .ood import OODDetection
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *

@register_pipeline_member
class EvaluateFeatureSpaceDistance(OODDetection):

    name = 'EvaluateFeatureSpaceDistance'

    def __init__(self, gpus=0, fit_to=[dconstants.TRAIN], p=2, k='all', layer=-2, **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.fit_to = fit_to
        self.p = p
        self.k = k
        self.layer = layer

    @property
    def configuration(self):
        return super().configuration | {
            'Fit to' : self.fit_to,
            'Norm' : self.p,
            'Num neighbours to consider' : self.k,
            'Features from layer' : self.layer,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        features_fit = [torch.cat(x, 0) for x in run_model_on_datasets(
            kwargs['model'], 
            [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], 
            gpus=self.gpus, model_kwargs=self.model_kwargs_fit,
            callbacks=[
                make_callback_get_features(layer=self.layer),
            ]
        )][0]
        features_eval, labels_eval = [torch.cat(x, 0) for x in run_model_on_datasets(
            kwargs['model'], 
            [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], 
            gpus=self.gpus, model_kwargs=self.model_kwargs_fit,
            callbacks=[
                make_callback_get_features(layer=self.layer),
                make_callback_get_ground_truth(),
            ]
        )]
        distances = torch.cdist(features_eval, features_fit, p=self.p) # [num_eval x num_train]
        distances_sorted, _ = torch.sort(distances, dim=-1)
        if self.k != 'all':
            distances_sorted = distances_sorted[:, :self.k]
        proxy = distances_sorted.mean(-1) # [num_eval]

        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)
        self.ood_detection(-torch.Tensor(proxy), labels_eval,
                                'feature-distance',
                                auroc_labels, auroc_mask, distribution_labels,
                                distribution_label_names, plot_proxy_log_scale=False, **kwargs
        )
        return args, kwargs
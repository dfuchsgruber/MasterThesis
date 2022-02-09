from evaluation.callbacks import make_callback_get_features, make_callback_get_ground_truth
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl


from .base import *
import data.constants as dconstants
from .ood import OODDetection
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *
from .feature_space_density import FeatureDensity

@register_pipeline_member
class EvaluateFeatureSpaceDistance(FeatureDensity):

    name = 'EvaluateFeatureSpaceDistance'

    def __init__(self, gpus=0, p=2, k='all', layer=-2, **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.p = p
        self.k = k
        self.layer = layer

    @property
    def configuration(self):
        return super().configuration | {
            'Norm' : self.p,
            'Num neighbours to consider' : self.k,
            'Features from layer' : self.layer,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        self._get_features_and_labels_to_fit(**kwargs)
        (features_to_fit, predictions_to_fit, labels_to_fit), (features_to_validate, prediction_to_validate, labels_to_validate) = self._get_features_and_labels_to_fit(**kwargs)
        features_to_evaluate, predictions_to_evaluate, labels_to_evaluate = self._get_features_and_labels_to_evaluate(**kwargs)
        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)


        torch.save({
            'features_fit' : features_to_fit,
            'predictions_fit' : predictions_to_fit,
            'labels_fit' : labels_to_fit,
            'features_eval' : features_to_evaluate,
            'predictions_eval' : predictions_to_evaluate,
            'labels_eval' : labels_to_evaluate,
        }, 'features_debug.pt')

        
        distances = torch.cdist(features_to_evaluate, features_to_fit, p=self.p) # [num_eval x num_train]
        distances_sorted, _ = torch.sort(distances, dim=-1)
        if self.k != 'all':
            distances_sorted = distances_sorted[:, :self.k]
        proxy = distances_sorted.mean(-1) # [num_eval]

        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)
        self.ood_detection(-torch.Tensor(proxy), labels_to_evaluate,
                                'feature-distance',
                                auroc_labels, auroc_mask, distribution_labels,
                                distribution_label_names, plot_proxy_log_scale=False, **kwargs
        )
        return args, kwargs
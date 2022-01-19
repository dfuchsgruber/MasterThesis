import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

from .base import *
import data.constants as dconstants
import evaluation.callbacks
from evaluation.util import run_model_on_datasets, get_data_loader
from data.util import vertex_intersection
from plot.util import plot_histograms, plot_2d_histogram
from evaluation.logging import *

@register_pipeline_member
class LogInductiveFeatureShift(PipelineMember):
    """ Logs the feature shift in vertices of the train-graph when re-introducing new edges of the val-graph. """

    name = 'LogInductiveFeatureShift'

    def __init__(self, gpus=0, data_before=dconstants.TRAIN, data_after=dconstants.OOD_VAL, eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.data_before = data_before
        self.data_after = data_after
        self.eps = eps

    @property
    def configuration(self):
        return super().configuration | {
            'Dataset before shift' : {self.data_before},
            'Dataset after shift' : {self.data_after},
            'Feature Shift epislon' : {self.eps},
        }
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        
        cfg: configuration.ExperimentConfiguration = kwargs['config']
        receptive_field_size = len(cfg.model.hidden_sizes) + 1
        callbacks = [
            evaluation.callbacks.make_callback_get_features(mask=False), 
            evaluation.callbacks.make_callback_get_data(),
        ]
        for k in range(1, receptive_field_size + 1):
            callbacks += [
                evaluation.callbacks.make_callback_count_neighbours_with_attribute(lambda data, output: ~data.is_out_of_distribution.numpy(), k, mask=False),
                evaluation.callbacks.make_callback_count_neighbours(k, mask=False),
            ]
        results = run_model_on_datasets(kwargs['model'], [get_data_loader(data, kwargs['data_loaders']) for data in (self.data_before, self.data_after)], gpus=self.gpus, callbacks=callbacks, model_kwargs=self.model_kwargs)
        features, data, num_nbs_id, num_nbs = results[0], results[1], results[2::2], results[3::2]

        idx_before, idx_after = vertex_intersection(data[0], data[1])
        shift = (features[0][idx_before] - features[1][idx_after]).norm(dim=1)
        
        # Log the feature shift by "in mask / not in mask"
        fig, ax = plot_histograms(shift.cpu() + self.eps, data[0].mask[idx_before].cpu(), 
                label_names={True : f'In {self.data_before}', False : f'Not in {self.data_before}'}, log_scale=True, kind='overlapping', x_label='Feature Shift')
        log_figure(kwargs['logs'], fig, f'feature_shift_by_mask', f'inductive_feature_shift{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
        plt.close(fig)
        pipeline_log(f'Logged inductive feature shift for data {self.data_before} -> {self.data_after} by mask.')

        for k in range(1, receptive_field_size + 1):
            fraction = 1 - (num_nbs_id[k - 1][1].float() / (num_nbs[k - 1][1] + 1e-12))
            fig, ax = plot_2d_histogram(shift.cpu() + self.eps, fraction[idx_after], x_label='Log Feature Shift', y_label=f'Fraction of ood vertices in {k} neighbourhood', log_scale_x=True)
            log_figure(kwargs['logs'], fig, f'feature_shift_by_{k}_nbs', f'inductive_feature_shift{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            plt.close(fig)
            pipeline_log(f'Logged inductive feature shift for data {self.data_before} -> {self.data_after} by {k}-hop neighbourhood.')
        
        return args, kwargs

@register_pipeline_member
class LogInductiveSoftmaxEntropyShift(PipelineMember):
    """ Logs the shift of softmax entropy in vertices of the train-graph when re-introducing new edges of the val-graph. """

    name = 'LogInductiveSoftmaxEntropyShift'

    def __init__(self, gpus=0, data_before=dconstants.TRAIN, data_after=dconstants.OOD_VAL, eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.data_before = data_before
        self.data_after = data_after
        self.eps = eps

    @property
    def configuration(self):
        return super().configuration | {
            'Dataset before shift' : {self.data_before},
            'Dataset after shift' : {self.data_after},
            'Entropy calculation epsilon' : {self.eps},
        }
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        
        cfg: configuration.ExperimentConfiguration = kwargs['config']
        receptive_field_size = len(cfg.model.hidden_sizes) + 1
        callbacks = [
            evaluation.callbacks.make_callback_get_predictions(mask=False, ensemble_average=False), # Average the prediction scores over the ensemble
            evaluation.callbacks.make_callback_get_data(),
        ]
        for k in range(1, receptive_field_size + 1):
            callbacks += [
                evaluation.callbacks.make_callback_count_neighbours_with_attribute(lambda data, output: ~data.is_out_of_distribution.numpy(), k, mask=False),
                evaluation.callbacks.make_callback_count_neighbours(k, mask=False),
            ]
        results = run_model_on_datasets(kwargs['model'], [get_data_loader(data, kwargs['data_loaders']) for data in (self.data_before, self.data_after)], gpus=self.gpus, callbacks=callbacks, model_kwargs=self.model_kwargs)
        scores, data, num_nbs_id, num_nbs = results[0], results[1], results[2::2], results[3::2]
        entropy = [
            -(score * torch.log2(score + self.eps)).sum(1).mean(-1) # Average over ensemble axis
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
            fraction = 1 - (num_nbs_id[k - 1][1].float() / (num_nbs[k - 1][1] + 1e-12))
            fig, ax = plot_2d_histogram(shift.cpu() + self.eps, fraction[idx_after], x_label='Entropy Shift', y_label=f'Fraction of ood vertices in {k} neighbourhood', log_scale_x=False)
            log_figure(kwargs['logs'], fig, f'entropy_shift_by_{k}_nbs', f'inductive_entropy_shift{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            plt.close(fig)
            pipeline_log(f'Logged inductive entropy shift for data {self.data_before} -> {self.data_after} by {k}-hop neighbourhood.')
        
        return args, kwargs


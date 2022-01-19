import torch
import matplotlib.pyplot as plt

from .base import *
import data.constants as dconstants
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *
import evaluation.callbacks
from metrics import expected_calibration_error
from plot.calibration import plot_calibration

@register_pipeline_member
class EvaluateCalibration(PipelineMember):
    """ Evaluates the calibration of a model. """

    name = 'EvaluateCalibration'

    def __init__(self, evaluate_on=[dconstants.VAL], bins=10, gpus=0, eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.evaluate_on = evaluate_on
        self.gpus = gpus
        self.bins = bins
        self.eps = eps

    @property
    def configuration(self):
        return super().configuration | {
            'Evaluate on' : self.evaluate_on,
            'Bins' : self.bins,
            'Epsilon' : self.eps,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        
        pred, gnd = run_model_on_datasets(kwargs['model'], [get_data_loader(dataset, kwargs['data_loaders']) for dataset in self.evaluate_on], gpus=self.gpus,
            callbacks = [
                evaluation.callbacks.make_callback_get_predictions(),
                evaluation.callbacks.make_callback_get_ground_truth(),
            ], model_kwargs=self.model_kwargs_evaluate)
        scores, gnd = torch.cat(pred, dim=0), torch.cat(gnd, dim=0)
        ece = expected_calibration_error(scores, gnd, bins=self.bins, eps=self.eps)

        dataset_names = '-'.join(self.evaluate_on)
        kwargs['metrics'][f'ece_{dataset_names}{self.suffix}'] = ece
        log_metrics(kwargs['logs'], {f'ece_{dataset_names}{self.suffix}' : ece}, f'calibration')

        if self.log_plots:
            fig, ax = plot_calibration(scores, gnd, bins=self.bins, eps=self.eps)
            log_figure(kwargs['logs'], fig, f'calibration_{dataset_names}{self.suffix}', f'calibration', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
            plt.close(fig)
        
        pipeline_log(f'Logged calibration for {self.evaluate_on}')
        return args, kwargs

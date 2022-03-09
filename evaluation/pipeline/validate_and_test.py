import torch
import pytorch_lightning as pl

from .base import *
import data.constants as dconstants
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *


@register_pipeline_member
class ValidateAndTest(PipelineMember):

    name = 'ValidateAndTest'

    """ Runs the test-step method of a pl module on different datasets. """
    def __init__(self, *args, evaluate_on=[dconstants.OOD_VAL], **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_on = evaluate_on

    @property
    def configuration(self):
        return super().configuration | {
            'evaluate_on' : self.evaluate_on,
        }

    def __call__(self, *args, **kwargs):
        model = kwargs['model']
        data_loaders = {name : get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on}
        for name, loader in data_loaders.items():
            metrics = model.step(loader.dataset[0], 0, prefix=f'pipeline_{name}{self.suffix}', log_metrics=False)
            for metric, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                kwargs['metrics'][f'{metric}_{name}{self.suffix}'] = v
                pipeline_log(f'{metric}_{name}{self.suffix} : {v}')

        return args, kwargs
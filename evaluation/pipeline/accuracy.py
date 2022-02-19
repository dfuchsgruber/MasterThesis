import torch

from .base import *
import data.constants as dconstants
from .uncertainty_quantification import OODSeparation
import evaluation.callbacks
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *

@register_pipeline_member
class EvaluateAccuracy(OODSeparation):
    """ Pipeline member to evaluate the accuracy of the model on a dataset. Note: The dataset should follow the train labels. """

    name = 'EvaluateAccuracy'

    def __init__(self, evaluate_on=[dconstants.OOD_VAL], gpus=0, **kwargs):
        super().__init__(evaluate_on=evaluate_on, **kwargs)
        self.gpus = gpus

    @property
    def configuration(self):
        return super().configuration
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        
        cfg: configuration.ExperimentConfiguration = kwargs['config']
        predictions, labels, mask = run_model_on_datasets(
            kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], 
            gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate,
            callbacks = [
                evaluation.callbacks.make_callback_get_predictions(),
                evaluation.callbacks.make_callback_get_ground_truth(),
                evaluation.callbacks.make_callback_is_ground_truth_in_labels(
                    cfg.data.train_labels),

            ])
        mask, predictions, labels = torch.cat(mask, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)
        is_id, is_id_mask, _, _ = self.get_ood_distribution_labels(**kwargs)

        # Accuracy should only be computed for classes the model can actually predict
        predictions, labels, is_id_mask, is_id = predictions[mask], labels[mask], is_id_mask[mask], is_id[mask]

        _, hard = predictions.max(dim=-1)
        acc = (hard == labels).float().mean()
        acc_id = (hard == labels)[(is_id == 1) & is_id_mask].float().mean()
        acc_ood = (hard == labels)[(is_id == 0) & is_id_mask].float().mean()

        dataset_names = '-'.join(self.evaluate_on)
        kwargs['metrics'][f'accuracy_{dataset_names}{self.suffix}'] = acc.item()
        kwargs['metrics'][f'accuracy_id_{dataset_names}{self.suffix}'] = acc_id.item()
        kwargs['metrics'][f'accuracy_ood_{dataset_names}{self.suffix}'] = acc_ood.item()
        pipeline_log(f'Evaluated accuracy for {self.evaluate_on}.')

        return args, kwargs


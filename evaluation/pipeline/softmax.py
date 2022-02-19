import torch

from .base import *
import data.constants as dconstants
from .uncertainty_quantification import UncertaintyQuantification
import evaluation.callbacks
from evaluation.util import run_model_on_datasets, get_data_loader

@register_pipeline_member
class EvaluateLogitEnergy(UncertaintyQuantification):
    """ Pipeline member to evaluate the Logit Energy curves of the model for in-distribution and out-of-distribution data. """

    name = 'EvaluateLogitEnergy'

    def __init__(self, gpus=0, evaluate_on=[dconstants.VAL], separate_distributions_by='ood-and-neighbourhood', 
                separate_distributions_tolerance=0.0, log_plots=True, temperature=1.0, **kwargs):
        super().__init__(separate_distributions_by=separate_distributions_by, 
                            separate_distributions_tolerance=separate_distributions_tolerance,
                            evaluate_on=evaluate_on,
                            **kwargs)
        self.gpus = gpus
        self.log_plots = log_plots
        self.temperature = temperature

    @property
    def configuration(self):
        return super().configuration | {
            'Log plots' : self.log_plots,
            'Temperature' : self.temperature,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        data_loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
        logits, labels, predictions = run_model_on_datasets(kwargs['model'], data_loaders, callbacks=[
                evaluation.callbacks.make_callback_get_logits(mask=True, ensemble_average=False),
                evaluation.callbacks.make_callback_get_ground_truth(mask=True),
                evaluation.callbacks.make_callback_get_predictions(mask=True, soft=True),
            ], gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate)
        logits, labels, predictions = torch.cat(logits), torch.cat(labels), torch.cat(predictions, dim=0) # Logits are of shape : N, n_classes, n_ensemble
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1) # N, n_ensemble
        energy = energy.mean(-1) # Average over ensemble members

        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_ood_distribution_labels(**kwargs)
        self.uncertainty_quantification(-energy, labels, 'logit-energy', predictions.argmax(1) == labels, auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
        
        return args, kwargs

@register_pipeline_member
class EvaluateSoftmaxEntropy(UncertaintyQuantification):
    """ Pipeline member to evaluate the Softmax Entropy curves of the model for in-distribution and out-of-distribution data. """

    name = 'EvaluateSoftmaxEntropy'

    def __init__(self, gpus=0, evaluate_on=[dconstants.VAL], separate_distributions_by='ood-and-neighbourhood', 
                separate_distributions_tolerance=0.0, log_plots=True, entropy_eps = 1e-12, variance_eps=1e-12, **kwargs):
        super().__init__(separate_distributions_by=separate_distributions_by, 
                            separate_distributions_tolerance=separate_distributions_tolerance,
                            evaluate_on=evaluate_on,
                            **kwargs)
        self.gpus = gpus
        self.log_plots = log_plots
        self.entropy_eps = entropy_eps
        self.variance_eps = variance_eps

    @property
    def configuration(self):
        return super().configuration | {
            'Log plots' : self.log_plots,
            'Epsilon for entropy calculation' : self.entropy_eps,
            'Epsilon for variance calculation' : self.variance_eps,
        }

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        data_loaders = [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on]
        scores, labels = run_model_on_datasets(kwargs['model'], data_loaders, callbacks=[
                evaluation.callbacks.make_callback_get_predictions(mask=True, ensemble_average=False), # Average the prediction scores over the ensemble
                evaluation.callbacks.make_callback_get_ground_truth(mask=True),
            ], gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate)
        scores, labels = torch.cat(scores), torch.cat(labels) # Scores are of shape : N, n_classes, n_ensemble

        is_correct_prediction = scores.mean(-1).argmax(1) == labels

        # Aleatoric uncertainty is the expected entropy
        expected_entropy = -(scores * torch.log2(scores + self.entropy_eps)).sum(1)
        expected_entropy = expected_entropy.mean(-1) # Get the expectation over all ensemble members

        # Epistemic uncertainty is the information gain, i.e. predictive uncertainty - aleatoric uncertainty
        avg_scores = scores.mean(-1) # Expectations of predictions in all ensemble members
        predictive_entropy = -(avg_scores * torch.log2(avg_scores + self.entropy_eps)).sum(1)

        max_scores, argmax_scores = scores.mean(-1).max(-1)

        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_ood_distribution_labels(**kwargs)
        self.uncertainty_quantification(-predictive_entropy, labels, 'total-predictive-entropy', is_correct_prediction, auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
        self.uncertainty_quantification(max_scores, labels, 'max-score', is_correct_prediction, auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
        if scores.size()[-1] > 1: # Some ensembling is used (ensembles, dropout, etc.), so epistemic and aleatoric estimates can be disentangled
            self.uncertainty_quantification(-expected_entropy, labels, 'expected-softmax-entropy', is_correct_prediction, auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
            self.uncertainty_quantification(-(predictive_entropy - expected_entropy), labels, 'mutual-information', is_correct_prediction, auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
            # Also use the empirical variance of the predicted class as proxy
            var = torch.var(scores, dim=-1) # Variance across ensemble, shape N x num_classes
            var_predicted_class = var[torch.arange(argmax_scores.size(0)), argmax_scores]
            self.uncertainty_quantification(1 / (var_predicted_class + self.variance_eps), labels, 'predicted-class-variance', is_correct_prediction, auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)
            var_total = var.sum(-1)
            self.uncertainty_quantification(1 / (var_total + self.variance_eps), labels, 'sampled-class-variance', is_correct_prediction, auroc_labels, auroc_mask, distribution_labels, distribution_label_names, plot_proxy_log_scale=False, log_plots=self.log_plots, **kwargs)

        return args, kwargs

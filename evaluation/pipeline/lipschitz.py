import numpy as np
import matplotlib.pyplot as plt
import torch

from evaluation.logging import *
from .base import *
from evaluation.util import get_data_loader
import plot.perturbations
from evaluation.lipschitz import *

@register_pipeline_member
class EvaluateEmpircalLowerLipschitzBounds(PipelineMember):
    """ Pipeline element for evaluation of Lipschitz bounds. """

    name = 'EvaluateEmpircalLowerLipschitzBounds'

    def __init__(self, num_perturbations_per_sample=5, min_perturbation=0.1, max_perturbation=5.0, num_perturbations = 10, 
            seed=None, gpus=0, evaluate_on=['val'], perturbation_type='noise', permute_per_sample=True, **kwargs):
        super().__init__(**kwargs)
        self.num_perturbations_per_sample = num_perturbations_per_sample
        self.min_perturbation = min_perturbation
        self.max_perturbation = max_perturbation
        self.num_perturbations = num_perturbations
        self.perturbation_type = perturbation_type
        self.permute_per_sample = permute_per_sample
        self.seed = seed
        self.gpus = gpus
        self.evaluate_on = evaluate_on

    @property
    def configuration(self):
        config = super().configuration
        config |= {
            'Perturbation type' : self.perturbation_type,
            'Number of perturbations per sample' : self.num_perturbations,
            'Seed' : self.seed,
            'Evaluate on' : self.evaluate_on,
        }
        if self.perturbation_type.lower() == 'noise':
            config['Min perturbation'] = self.min_perturbation
            config['Max perturbation'] = self.max_perturbation
        elif self.perturbation_type.lower() == 'derangement':
            config['Permutation per Sample'] = self.permute_per_sample
        return config

    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        for name in self.evaluate_on:
            loader = get_data_loader(name, kwargs['data_loaders'])
            assert len(loader) == 1, f'Empirical local lipschitz evaluation is currently only supported for semi-supervised tasks.'
            for dataset in loader:
                break # Just want the single dataset
            if self.gpus > 0:
                dataset = dataset.to('cuda')
                kwargs['model'] = kwargs['model'].to('cuda')
            if self.perturbation_type.lower() == 'noise':
                perturbations = local_perturbations(kwargs['model'], dataset,
                    perturbations=np.linspace(self.min_perturbation, self.max_perturbation, self.num_perturbations),
                    num_perturbations_per_sample=self.num_perturbations_per_sample, seed=self.seed, model_kwargs=self.model_kwargs)
                pipeline_log(f'Created {self.num_perturbations_per_sample} perturbations in linspace({self.min_perturbation:.2f}, {self.max_perturbation:.2f}, {self.num_perturbations}) for {name} samples.')
            elif self.perturbation_type.lower() == 'derangement':
                perturbations = permutation_perturbations(kwargs['model'], dataset,
                    self.num_perturbations, num_perturbations_per_sample=self.num_perturbations_per_sample, 
                    seed=self.seed, per_sample=self.permute_per_sample, model_kwargs=self.model_kwargs)
                pipeline_log(f'Created {self.num_perturbations} permutations ({self.num_perturbations_per_sample} each) for {name} samples.')

            smean, smedian, smax, smin = local_lipschitz_bounds(perturbations)
            metrics = {
                f'{name}_slope_mean_perturbation' : smean,
                f'{name}_slope_median_perturbation' : smedian,
                f'{name}_slope_max_perturbation' : smax,
                f'{name}_slope_min_perturbation' : smin,
            }
            log_metrics(kwargs['logs'], metrics, f'empirical_lipschitz{self.suffix}')
            for metric, value in metrics.items():
                kwargs['metrics'][f'empirical_lipschitz_{metric}{self.suffix}'] = value
            # Plot the perturbations and log it
            if self.log_plots:
                fig, _ , _, _ = plot.perturbations.local_perturbations_plot(perturbations)
                log_figure(kwargs['logs'], fig, f'{name}_perturbations', f'empirical_lipschitz{self.suffix}', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Logged input vs. output perturbation plot for dataset {name}.')
                plt.close(fig)
        
        return args, kwargs



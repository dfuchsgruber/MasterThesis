import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from util import is_outlier


from .base import *
import data.constants as dconstants
from .ood import OODDetection
import evaluation.callbacks
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *
import evaluation.constants as econstants
from data.util import label_binarize
from model.dimensionality_reduction import DimensionalityReduction
from model.density import get_density_model
from plot.density import plot_density, get_dimensionality_reduction_to_plot
from plot.util import get_greyscale_colormap

class FeatureDensity(OODDetection):
    """ Superclass for pipeline members that fit a feature density. """

    name = 'FeatureDensity'

    def __init__(self, gpus=0, fit_to=[dconstants.TRAIN], 
        fit_to_ground_truth_labels=[dconstants.TRAIN], fit_to_mask_only=True, fit_to_best_prediction=False,
        fit_to_min_confidence = 0.0,
        **kwargs):
        super().__init__(
            **kwargs
        )
        self.gpus = gpus
        self.fit_to = fit_to
        self.fit_to_ground_truth_labels = fit_to_ground_truth_labels
        self.fit_to_mask_only = fit_to_mask_only
        self.fit_to_best_prediction = fit_to_best_prediction
        self.fit_to_min_confidence = fit_to_min_confidence

    @property
    def configuration(self):
        return super().configuration | {
            'Fit to' : self.fit_to,
            'Fit to mask only' : self.fit_to_mask_only,
            'Use best prediction only' : self.fit_to_best_prediction,
            'Use predictions with minimal confidence only' : self.fit_to_min_confidence,
            'Use ground truth labels for fit on' : self.fit_to_ground_truth_labels,
        }

    def _get_features_and_labels_to_fit(self, **kwargs):
        features, predictions, labels, mask = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.fit_to], gpus=self.gpus, model_kwargs=self.model_kwargs_fit,
            callbacks = [
                evaluation.callbacks.make_callback_get_features(mask = self.fit_to_mask_only),
                evaluation.callbacks.make_callback_get_predictions(mask = self.fit_to_mask_only),
                evaluation.callbacks.make_callback_get_ground_truth(mask = self.fit_to_mask_only),
                evaluation.callbacks.make_callback_get_mask(mask = self.fit_to_mask_only),
            ])
        for idx, name in enumerate(self.fit_to):
            if name.lower() in self.fit_to_ground_truth_labels: # Override predictions with ground truth for training data, but only within the mask
                predictions[idx][mask[idx]] = label_binarize(labels[idx][mask[idx]], num_classes=predictions[idx].size(1)).float()
        features, predictions, labels, mask = torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels), torch.cat(mask)
        if self.fit_to_best_prediction:
            predictions *= label_binarize(predictions.argmax(1)) # Mask such that only the best prediction is remaining
        predictions[predictions < self.fit_to_min_confidence] = 0.0

        return features, predictions, labels

    def _get_features_and_labels_to_evaluate(self, **kwargs):
        features, predictions, labels = run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate)
        return torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)

@register_pipeline_member
class FitFeatureDensityGrid(FeatureDensity):
    """ Pipeline member that fits a grid of several densities to the feature space of a model. """

    name = 'FitFeatureDensityGrid'

    def __init__(self, density_types={}, dimensionality_reductions={}, seed=1337,
                    density_plots = ['pca', 'umap'],
                    **kwargs):
        super().__init__(**kwargs)
        self.density_types = density_types
        self.dimensionality_reductions = dimensionality_reductions
        self.seed = seed
        self.density_plots = density_plots

    @property
    def configuration(self):
        return super().configuration | {
            'Density types' : self.density_types,
            'Dimensionality Reductions' : self.dimensionality_reductions,
            'Density Plots' : self.density_plots,
            'Seed' : self.seed,
        }
        
    @torch.no_grad()
    def __call__(self, *args, **kwargs):

        if self.seed is not None:
            pl.seed_everything(self.seed)

        # Only compute data once
        features_to_fit, predictions_to_fit, labels_to_fit = self._get_features_and_labels_to_fit(**kwargs)
        # print(features_to_fit.size(), predictions_to_fit)

        # Note that for `self.fit_to_ground_truth_labels` data, the `predictions_to_fit` are overriden with a 1-hot ground truth
        features_to_evaluate, predictions_to_evaluate, labels_to_evaluate = self._get_features_and_labels_to_evaluate(**kwargs)
        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_distribution_labels(**kwargs)

        # Grid over dimensionality reductions
        for dim_reduction_type, dim_reduction_grid in self.dimensionality_reductions.items():
            keys = list(dim_reduction_grid.keys())
            for values in product(*[dim_reduction_grid[k] for k in keys]):
                dim_reduction_config = {key : values[idx] for idx, key in enumerate(keys)}
                dim_reduction = DimensionalityReduction(type=dim_reduction_type, per_class=False, **dim_reduction_config)
                dim_reduction.fit(features_to_fit)
                pipeline_log(f'{self.name} fitted dimensionality reduction {dim_reduction.compressed_name}')

                # TODO: dim-reductions transform per-class, but this is not supported here, so we just take any and set `per_class` to false in its constructor
                features_to_fit_reduced = torch.from_numpy(dim_reduction.transform(features_to_fit))
                features_to_evaluate_reduced = torch.from_numpy(dim_reduction.transform(features_to_evaluate))

                # Get a 2d projection for plotting
                if self.log_plots:
                    embeddings, is_train, grids, grid_inverses = get_dimensionality_reduction_to_plot(features_to_fit.numpy(),
                        features_to_evaluate_reduced.numpy(), self.density_plots)
                    pipeline_log(f'Created plotting grids for reduction {dim_reduction.compressed_name}')

                # Grid over feature space densities
                for density_type, density_grid in self.density_types.items():
                    keys_density = list(density_grid.keys())
                    for values_density in product(*[density_grid[k] for k in keys_density]):
                        density_config = {key : values_density[idx] for idx, key in enumerate(keys_density)}
                        density_model = get_density_model(
                            density_type=density_type, 
                            **density_config,
                            )
                        density_model.fit(features_to_fit_reduced, predictions_to_fit)
                        pipeline_log(f'{self.name} fitted density {density_model.compressed_name}')

                        for eval_suffix, eval_kwargs in density_model.evaluation_kwargs:
                            log_density = density_model(features_to_evaluate_reduced, **eval_kwargs).cpu()
                            is_finite_density = torch.isfinite(log_density)
                            proxy_name = f'{density_model.compressed_name}{eval_suffix}:{dim_reduction.compressed_name}'
                            self.ood_detection(log_density[is_finite_density], labels_to_evaluate[is_finite_density],
                                proxy_name,
                                auroc_labels[is_finite_density], auroc_mask[is_finite_density], distribution_labels[is_finite_density],
                                distribution_label_names, plot_proxy_log_scale=False, **kwargs
                            )
                            if self.log_plots:
                                # Get bounds for the heatmap values
                                log_density_fit = density_model(features_to_fit_reduced, **eval_kwargs).cpu()
                                log_density_data = torch.cat([log_density_fit, log_density], 0).numpy()
                                log_density_data = log_density_data[~is_outlier(log_density_data)]
                                vmin, vmax = log_density_data.min(), log_density_data.max()
                                vmin, vmax = vmin - 0.5 * (vmax - vmin), vmax + 0.5 * (vmax - vmin)
                                for plotting_type in self.density_plots:
                                    bins_x, bins_y = grids[plotting_type].shape[0], grids[plotting_type].shape[1]
                                    density_grid = density_model(
                                        torch.Tensor(grid_inverses[plotting_type].reshape(bins_x * bins_y, -1)), **eval_kwargs).cpu().numpy().reshape((bins_x, bins_y))

                                    embeddings_fit = embeddings[plotting_type][is_train]
                                    embeddings_eval = embeddings[plotting_type][~is_train] 

                                    fig, ax = plot_density(embeddings_fit, embeddings_eval[is_finite_density], grids[plotting_type], density_grid, 
                                        distribution_labels[is_finite_density].numpy(), distribution_label_names, cmap=get_greyscale_colormap(), vmin=vmin, vmax=vmax, colors=econstants.DISTRIBUTION_COLORS)
                                    log_figure(kwargs['logs'], fig, f'{plotting_type}{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                                    plt.close(fig)

        return args, kwargs

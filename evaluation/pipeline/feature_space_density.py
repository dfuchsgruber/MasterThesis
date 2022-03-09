import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from util import is_outlier


from .base import *
import data.constants as dconstants
from .uncertainty_quantification import UncertaintyQuantification
import evaluation.callbacks
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *
import evaluation.constants as econstants
from data.util import label_binarize
from model.dimensionality_reduction import DimensionalityReduction
from model.density import get_density_model
from plot.density import plot_density, get_dimensionality_reduction_to_plot
from plot.util import get_greyscale_colormap
from util import approximate_page_rank_matrix

class FeatureDensity(UncertaintyQuantification):
    """ Superclass for pipeline members that fit a feature density. """

    name = 'FeatureDensity'

    def __init__(self, gpus=0, fit_to=[dconstants.TRAIN], 
        fit_to_ground_truth_labels=[dconstants.TRAIN, dconstants.VAL], validate_on=[dconstants.VAL], fit_to_mask_only=True, fit_to_best_prediction=False,
        fit_to_min_confidence = 0.0, 
        diffuse_features = False, diffusion_iterations = 16, teleportation_probability = 0.2,
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
        self.validate_on = validate_on
        self.diffuse_features = diffuse_features
        self.diffusion_iterations = diffusion_iterations
        self.teleportation_probability = teleportation_probability

    @property
    def configuration(self):
        return super().configuration | {
            'Fit to' : self.fit_to,
            'Fit to mask only' : self.fit_to_mask_only,
            'Use best prediction only' : self.fit_to_best_prediction,
            'Use predictions with minimal confidence only' : self.fit_to_min_confidence,
            'Use ground truth labels for fit on' : self.fit_to_ground_truth_labels,
            'Validate on' : self.validate_on,
            'Diffuse features' : self.diffuse_features,
            'Diffusion iterations' : self.diffusion_iterations,
            'Teleportation probability' : self.teleportation_probability,
        }

    def _get_features_and_labels_to_fit(self, **kwargs):
        """ Gets features and labels used for density fitting and validation. """
        results = []
        for datasets in (self.fit_to, self.validate_on):
            features, predictions, labels = [], [], []
            for x, pred, y, mask, edge_index, name in zip(*run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in datasets], 
                gpus=self.gpus, model_kwargs=self.model_kwargs_fit, callbacks = [
                    evaluation.callbacks.make_callback_get_features(mask = False),
                    evaluation.callbacks.make_callback_get_predictions(mask = False),
                    evaluation.callbacks.make_callback_get_ground_truth(mask = False),
                    evaluation.callbacks.make_callback_get_mask(mask = False),
                    evaluation.callbacks.make_callback_get_attribute(lambda data, output: data.edge_index, mask=False)  
                ]), datasets):
                if self.diffuse_features:
                    x = torch.matmul(torch.Tensor(approximate_page_rank_matrix(edge_index.numpy(), x.size(0),
                        diffusion_iterations=self.diffusion_iterations, alpha = self.teleportation_probability)), x)
                if name.lower() in self.fit_to_ground_truth_labels:
                    # Replace confidence values with 1.0 for the ground truth labels
                    pred[mask] = label_binarize(y[mask], num_classes=pred.size(1)).float()
                if self.fit_to_best_prediction:
                    pred *= label_binarize(pred.argmax(1)) # Zero out the not most-confident prediction
                pred[pred < self.fit_to_min_confidence] = 0.0 # Zero out predictions that are not confident enough
                if self.fit_to_mask_only:
                    x, pred, y = x[mask], pred[mask], y[mask]
                features.append(x)
                predictions.append(pred)
                labels.append(y)

            results.append((torch.cat(features, dim=0), torch.cat(predictions, dim=0), torch.cat(labels)))

        return results[0], results[1]
    

    def _get_features_and_labels_to_evaluate(self, **kwargs):

        features, predictions, labels = [], [], []
        
        for x, pred, y, mask, edge_index in zip(*run_model_on_datasets(kwargs['model'], [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], 
            gpus=self.gpus, model_kwargs=self.model_kwargs_evaluate, callbacks = [
                evaluation.callbacks.make_callback_get_features(mask = False),
                evaluation.callbacks.make_callback_get_predictions(mask = False),
                evaluation.callbacks.make_callback_get_ground_truth(mask = False),
                evaluation.callbacks.make_callback_get_mask(mask = False),
                evaluation.callbacks.make_callback_get_attribute(lambda data, output: data.edge_index, mask=False)  
            ])):
            if self.diffuse_features:
                x = torch.matmul(torch.Tensor(approximate_page_rank_matrix(edge_index.numpy(), x.size(0),
                    diffusion_iterations=self.diffusion_iterations, alpha = self.teleportation_probability)), x)
            features.append(x[mask])
            predictions.append(pred[mask])
            labels.append(y[mask])

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
        (features_to_fit, predictions_to_fit, labels_to_fit), (features_to_validate, predictions_to_validate, labels_to_validate) = self._get_features_and_labels_to_fit(**kwargs)
        # print(features_to_fit.size(), predictions_to_fit)

        # Note that for `self.fit_to_ground_truth_labels` data, the `predictions_to_fit` are overriden with a 1-hot ground truth
        features_to_evaluate, predictions_to_evaluate, labels_to_evaluate = self._get_features_and_labels_to_evaluate(**kwargs)
        auroc_labels, auroc_mask, distribution_labels, distribution_label_names = self.get_ood_distribution_labels(**kwargs)
        is_correct_prediction = predictions_to_evaluate.argmax(-1) == labels_to_evaluate

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
                features_to_validate_reduced = torch.from_numpy(dim_reduction.transform(features_to_validate))
                features_to_evaluate_reduced = torch.from_numpy(dim_reduction.transform(features_to_evaluate))

                # Get a 2d projection for plotting
                if self.log_plots:
                    embeddings, is_fit_or_val, grids, grid_inverses = get_dimensionality_reduction_to_plot(
                        torch.cat((features_to_fit, features_to_validate), 0).numpy(),
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
                        density_model.fit(features_to_fit_reduced, predictions_to_fit, features_to_validate_reduced, predictions_to_validate)
                        pipeline_log(f'{self.name} fitted density {density_model.compressed_name}')

                        for eval_suffix, eval_kwargs in density_model.evaluation_kwargs:
                            log_density = density_model(features_to_evaluate_reduced, **eval_kwargs).cpu()
                            is_finite_density = torch.isfinite(log_density)
                            proxy_name = f'{density_model.compressed_name}{eval_suffix}:{dim_reduction.compressed_name}'
                            self.uncertainty_quantification(log_density[is_finite_density], labels_to_evaluate[is_finite_density],
                                proxy_name, is_correct_prediction,
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

                                    embeddings_fit = embeddings[plotting_type][is_fit_or_val]
                                    embeddings_eval = embeddings[plotting_type][~is_fit_or_val] 

                                    fig, ax = plot_density(embeddings_fit, embeddings_eval[is_finite_density], grids[plotting_type], density_grid, 
                                        distribution_labels[is_finite_density].numpy(), distribution_label_names, cmap=get_greyscale_colormap(invert=True), vmin=vmin, vmax=vmax, 
                                        colors=econstants.DISTRIBUTION_COLORS, legend_labels={
                                            econstants.ID_CLASS_NO_OOD_CLASS_NBS : 'In distribution',
                                            econstants.OOD_CLASS_NO_ID_CLASS_NBS : 'Out of distribution',
                                        }, color_fit=econstants.COLOR_FIT,
                                        legend_x=1.09)
                                    log_figure(kwargs['logs'], fig, f'{plotting_type}{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                                    plt.close(fig)

                        # Manually free cuda memory
                        del density_model
                        torch.cuda.empty_cache()


        return args, kwargs

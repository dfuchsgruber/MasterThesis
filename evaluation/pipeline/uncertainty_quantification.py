from typing import Dict, Optional
import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import logging

from .base import *
import data.constants as dconstants
import evaluation.constants as econstants
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *
import evaluation.callbacks
from plot.util import plot_histograms
from plot.neighbours import plot_against_neighbourhood
import configuration

class OODSeparation(PipelineMember):
    """ Base class to perform any kind of method that separates id from ood data. """

    name = 'OODSeparation'
    
    def __init__(self, *args, separate_distributions_by='ood', separate_distributions_tolerance=0.0,
            evaluate_on=[dconstants.OOD_VAL], **kwargs):
        super().__init__(*args, **kwargs)
        self.separate_distributions_by = separate_distributions_by
        self.separate_distributions_tolerance = separate_distributions_tolerance
        self.evaluate_on = evaluate_on

    @property
    def configuration(self):
        return super().configuration | {
            'Evaluate on' : self.evaluate_on,
            'Separate distributions by' : self.separate_distributions_by,
            'Separate distributions tolerance' : self.separate_distributions_tolerance,
        }

    def _get_degree(self, mask=True, k = None, **kwargs):
        """ Gets the node degree in each k-hop neighbourhood.

        Parameters:
        -----------
        mask : bool
            If given, only vertices in the masks of the `self.evaluate_on` datasets are used.
        k : int or None
            Consider k-hop neighbourhoods in 0, 1, ... k
            If None is given, k is set to the receptive field of the underlying model configuration
        
        Returns:
        --------
        degree : torch.Tensor, shape [N, k + 1]
            Node degree for each k hop neighbourhood.
        """
        if k is None:
            cfg: configuration.ExperimentConfiguration = kwargs['config']
            k = len(cfg.model.hidden_sizes) + 1
        callbacks = [
            evaluation.callbacks.make_callback_get_degree(hops=i, mask=mask, cpu=True) for i in range(k + 1)
        ]
        return torch.stack([torch.cat(r, dim=0) for r in run_model_on_datasets(None, [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on], 
            callbacks=callbacks, run_model=False)], dim=-1)

    def _count_id_nbs(self, mask: bool=True, k: Optional[int] = None, fraction: bool=True, **kwargs) -> torch.Tensor:
        """ Counts the number of id neighbours in a set of given k-hop neighbourhoods..

        Parameters:
        -----------
        mask : bool
            If given, only vertices in the masks of the `self.evaluate_on` datasets are used.
        k : int or None
            Consider k-hop neighbourhoods in 0, 1, ... k
            If None is given, k is set to the receptive field of the underlying model configuration
        fraction : bool, optional, default: True
            If the count is to be normalized by the number of all neighbours in each k-hop neighbourhood.
        
        Returns:
        --------
        fraction_id_nbs : torch.Tensor, shape [N, k + 1]
            The fraction of id neighbours for each vertex in a corresponding k-neighbourhood.
        """ 
        if k is None:
            cfg: configuration.ExperimentConfiguration = kwargs['config']
            k = len(cfg.model.hidden_sizes) + 1
        fraction_id_nbs = evaluation.util.count_id_neighbours(
            [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on],
            k,
            mask=mask,
            fraction = fraction,
        )
        return fraction_id_nbs

    

    def _get_ood_distribution_labels_perturbations(self, mask=True, **kwargs):
        """ Gets labels for id vs ood where ood data is perturbed.
        
        Returns:
        --------
        ood_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for auroc calculation.
        ood_mask : torch.Tensor, shape [N]
            Which samples should be used for AUROC calculation.
        ood_distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions.
        ood_distribution_label_names : dict
            Mapping that names all the labels in `ood_distribution_labels`.
        """

        is_ood = run_model_on_datasets(None, 
            [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on],
            run_model = False,
            callbacks = [
                evaluation.callbacks.make_callback_count_neighbours_with_attribute(
                    lambda data, outputs: data.is_out_of_distribution.numpy(), 0, mask=mask,
                )
            ]
        )[0]
        is_ood = torch.cat(is_ood, dim=0)[:, 0]
        ood_distribution_labels = torch.zeros(is_ood.size(0))
        ood_distribution_labels[is_ood > 0] = econstants.OOD_CLASS_NO_ID_CLASS_NBS
        ood_distribution_labels[~(is_ood > 0)] = econstants.ID_CLASS_NO_OOD_CLASS_NBS
        ood_mask = torch.ones_like(ood_distribution_labels).bool()
        ood_labels = torch.zeros_like(ood_distribution_labels).bool()
        ood_labels[ood_distribution_labels == econstants.ID_CLASS_NO_OOD_CLASS_NBS] = True
        ood_distribution_label_names = {
            econstants.ID_CLASS_NO_OOD_CLASS_NBS : f'Unperturbed', 
            econstants.OOD_CLASS_NO_ID_CLASS_NBS : f'Perturbed', 
        }
        return ood_labels, ood_mask, ood_distribution_labels, ood_distribution_label_names

    def _get_ood_distribution_labels_leave_out_classes(self, mask=True, **kwargs):
        """ Gets labels for id vs ood where ood data is left out classes.
        
        Returns:
        --------
        ood_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for auroc calculation.
        ood_mask : torch.Tensor, shape [N]
            Which samples should be used for AUROC calculation.
        ood_distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions.
        ood_distribution_label_names : dict
            Mapping that names all the labels in `ood_distribution_labels`.
        """
        fraction_id_nbs = self._count_id_nbs(mask=mask, k=None, fraction=True, **kwargs)
        # print(f'Got fraction id nbs.')
        ood_distribution_labels = evaluation.util.get_ood_distribution_labels(fraction_id_nbs,threshold = self.separate_distributions_tolerance,)
        ood_labels, ood_mask = evaluation.util.separate_distributions(ood_distribution_labels, self.separate_distributions_by)
        ood_distribution_label_names = {
            econstants.OOD_CLASS_NO_ID_CLASS_NBS : f'OOD class, no ID class\nneighbours in {fraction_id_nbs.size(1) - 1} hops', 
            econstants.OOD_CLASS_ID_CLASS_NBS : f'OOD class, ID class\nneighbours in {fraction_id_nbs.size(1) - 1} hops',
            econstants.ID_CLASS_NO_OOD_CLASS_NBS : f'ID class, no OOD class\nneighbours in {fraction_id_nbs.size(1) - 1} hops', 
            econstants.ID_CLASS_ODD_CLASS_NBS : f'ID class, OOD class\nneighbours in {fraction_id_nbs.size(1) - 1} hops',
        }
        return ood_labels, ood_mask, ood_distribution_labels, ood_distribution_label_names

    def get_ood_distribution_labels(self, mask=True, **kwargs):
        """ Gets labels for id vs ood where ood data is left out classes.

        Parameters:
        -----------
        kind : str
            With which method to get out of distribution data.
        
        Returns:
        --------
        ood_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for auroc calculation.
        ood_mask : torch.Tensor, shape [N]
            Which samples should be used for AUROC calculation.
        ood_distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions.
        ood_distribution_label_names : dict
            Mapping that names all the labels in `ood_distribution_labels`.
        """
        cfg: configuration.ExperimentConfiguration = kwargs['config']
        if cfg.data.ood_type == dconstants.LEFT_OUT_CLASSES:
            return self._get_ood_distribution_labels_leave_out_classes(mask=mask, **kwargs)
        elif cfg.data.ood_type == dconstants.PERTURBATION:
            return self._get_ood_distribution_labels_perturbations(mask=mask, **kwargs)
        else:
            raise RuntimeError(f'Could not separate distribution labels (id vs ood) by unknown type {cfg.data.ood_type}.')

class UncertaintyQuantification(OODSeparation):
    """ Pipeline member to perform OOD detection for a given metric. Evaluates AUROC scores and logs plots. """

    name = 'UncertaintyQuantification'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def uncertainty_quantification(self, proxy: Tensor, labels: Tensor, proxy_name: str, is_correctly_classified: Tensor, 
        ood_labels: Tensor, ood_mask: Tensor, ood_distribution_labels: Tensor, ood_distribution_label_names: Dict[str, int],
                        plot_proxy_log_scale: bool=True,**kwargs):
        """ Performs uncertainty quantification (ood detection, misclassification detection, etc.) and logs metrics and plots.
        
        Parameters:
        -----------
        proxy : torch.Tensor, shape [N]
            The proxy for separating id and ood. Higher values should be assigned to id data.
        labels : torch.Tensor, shape [N]
            Ground truth labels. Used to separate the proxy values by ground truth label.
        proxy_name : str
            Name of the proxy to use.
        is_correctly_classified : Tensor, shape [N]
            If a sample is correctly classified by the model.
        ood_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for ood detection.
        ood_mask : torch.Tensor, shape [N]
            Which samples should be used for ood detection.
        ood_distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions in ood detection.
        ood_distribution_label_names : dict
            Mapping that names all the labels in `ood_distribution_labels`.
        plot_proxy_log_scale : bool, optional, default: True
            If `True`, the proxy will be plotted in log scale.
        """

        if plot_proxy_log_scale:
            proxy += 1e-10 # To be able to plot

        # Calculate area under the ROC for separating in-distribution (label 1) from out of distribution (label 0)
        roc_auc = roc_auc_score(ood_labels[ood_mask].cpu().long().numpy(), proxy[ood_mask].cpu().numpy()) # higher proxy -> higher confidence
        kwargs['metrics'][f'ood_auroc_{proxy_name}{self.suffix}'] = roc_auc
        log_metrics(kwargs['logs'], {f'ood_auroc_{proxy_name}{self.suffix}' : roc_auc}, f'{proxy_name}_plots')
        logging.info(f'ood_auroc_{proxy_name}{self.suffix} : {roc_auc}')

        # Calculate the area under the PR curve separating in-distribution (label 1) from out of distribution (label 0)
        precision, recall, _ = precision_recall_curve(ood_labels[ood_mask].cpu().long().numpy(), proxy[ood_mask].cpu().numpy())
        aucpr = auc(recall, precision)
        kwargs['metrics'][f'ood_aucpr_{proxy_name}{self.suffix}'] = aucpr
        log_metrics(kwargs['logs'], {f'ood_aucpr_{proxy_name}{self.suffix}' : aucpr}, f'{proxy_name}_plots')

        # Calculate area under the ROC for finding correctly classified instances
        misclassification_auroc = roc_auc_score(is_correctly_classified.cpu().long().numpy(), proxy.cpu().numpy()) 
        kwargs['metrics'][f'misclassification_auroc_{proxy_name}{self.suffix}'] = misclassification_auroc
        log_metrics(kwargs['logs'], {f'misclassification_auroc_{proxy_name}{self.suffix}' : misclassification_auroc}, f'{proxy_name}_plots')
        logging.info(f'misclassification_auroc_{proxy_name}{self.suffix} : {misclassification_auroc}')
        logging.info(f'aucpr_{proxy_name}{self.suffix} : {aucpr}')

        # Calculate the area under the PR curve separating in-distribution (label 1) from out of distribution (label 0)
        misclassification_precision, misclassification_recall, _ = precision_recall_curve(is_correctly_classified.cpu().long().numpy(), proxy.cpu().numpy())
        misclassification_aucpr = auc(misclassification_recall, misclassification_precision)
        kwargs['metrics'][f'misclassification_aucpr_{proxy_name}{self.suffix}'] = misclassification_aucpr
        log_metrics(kwargs['logs'], {f'misclassification_aucpr_{proxy_name}{self.suffix}' : misclassification_aucpr}, f'{proxy_name}_plots')

        # -------------- Different plots for an ood-detection proxy -------------

        # Proxy vs node degree in a k-hop neighbourhood
        try:
            if self.log_plots:
                deg = self._get_degree(mask=True, k=None, **kwargs).long()
                fig, axs = plot_against_neighbourhood(deg[ood_mask], proxy, ood_labels[ood_mask], x_label='Degree', y_label='Proxy', y_log_scale=plot_proxy_log_scale, k_min=1)
                log_figure(kwargs['logs'], fig, f'{proxy_name}_by_degree{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Saved {proxy_name} by degree to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_by_degree{self.suffix}.pdf')))
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not create vs. degree plots for {proxy_name}. Reason: {e}')

        # Distribution of proxy per class label
        try:
            if self.log_plots:
                y = labels.cpu()
                for label in torch.unique(y):
                    proxy_label = proxy[y == label]
                    log_histogram(kwargs['logs'], proxy_label.cpu().numpy(), f'{proxy_name}', global_step=label, label_suffix=str(label.item()))
                    log_metrics(kwargs['logs'], {
                        f'{self.prefix}mean_{proxy_name}' : proxy_label.mean(),
                        f'{self.prefix}std_{proxy_name}' : proxy_label.std(),
                        f'{self.prefix}min_{proxy_name}' : proxy_label.min(),
                        f'{self.prefix}max_{proxy_name}' : proxy_label.max(),
                        f'{self.prefix}median_{proxy_name}' : proxy_label.median(),
                    }, f'{proxy_name}_statistics', step=label)
                fig, ax = plot_histograms(proxy.cpu(), y.cpu(), log_scale=plot_proxy_log_scale, kind='vertical', x_label=f'Proxy', y_label='Class')
                log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_all_classes{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Evaluated {proxy_name}.')
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not create class label-wise plots for {proxy_name}.')

        # Distribution of proxy per distribution label (id with pure id neighbours, id with non-pure id neighbours, ...)
        try:
            if self.log_plots:
                fig, ax = plot_histograms(proxy.cpu(), ood_distribution_labels.cpu(), 
                    label_names=ood_distribution_label_names,
                    kind='vertical', kde=True, log_scale=plot_proxy_log_scale,  x_label=f'Proxy', y_label='Kind')
                log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_all_kinds{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Saved {proxy_name} (all kinds) histogram to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_histograms_all_kinds{self.suffix}.pdf')))
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not distribution label-wise plots for {proxy_name}.')

        # Distribution of proxy for id and ood data that are considered for AUROC calculation (i.e. are in ood_mask)
        try:
            if self.log_plots:
                fig, ax = plot_histograms(proxy[ood_mask].cpu(), ood_labels[ood_mask].cpu().long(), 
                    label_names={0 : 'Out ouf distribution', 1 : 'In distribution'},
                    kind='overlapping', kde=True, log_scale=plot_proxy_log_scale,  x_label=f'Proxy', y_label='Kind')
                log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_id_vs_ood{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Saved {proxy_name} histogram (id vs ood) to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_histograms_id_vs_ood{self.suffix}.pdf')))
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not id vs ood plots for {proxy_name}. Reason {e}')

        # Neighbourhood purity w.r.t. id neighbours for different hops
        # a) fraction of id neighbours
        try:
            if self.log_plots:
                fraction_id_nbs = self._count_id_nbs(mask=True, k=None, fraction=True, **kwargs)
                fig, axs = plot_against_neighbourhood(fraction_id_nbs[ood_mask], proxy, ood_labels[ood_mask], x_label='Fraction of in distirubtion neighbours', y_label='Proxy', y_log_scale=plot_proxy_log_scale, k_min=1, x_min=0.0, x_max=1.0)
                log_figure(kwargs['logs'], fig, f'{proxy_name}_by_fraction_id_nbs{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Saved {proxy_name} by fraction of id nbs to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_by_fraction_id_nbs{self.suffix}.pdf')))
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not plot by fraction of id nbs for {proxy_name}. Reason {e}')

        # b) number of all id neighbours
        try:
            if self.log_plots:
                num_id_nbs = self._count_id_nbs(mask=True, k=None, fraction=False, **kwargs)
                fig, axs = plot_against_neighbourhood(num_id_nbs[ood_mask], proxy, ood_labels[ood_mask], x_label='Number of in distirubtion neighbours', y_label='Proxy', y_log_scale=plot_proxy_log_scale, k_min=1)
                log_figure(kwargs['logs'], fig, f'{proxy_name}_by_num_id_nbs{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Saved {proxy_name} by fraction of id nbs to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_by_num_id_nbs{self.suffix}.pdf')))
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not plot by fraction of id nbs for {proxy_name}. Reason {e}')


        # Distribution of proxy for correctly and wrongly classified
        try:
            if self.log_plots:
                fig, ax = plot_histograms(proxy.cpu(), is_correctly_classified.long().cpu(), 
                    label_names={0 : 'Misclassified', 1 : 'Correctly Classified'},
                    kind='overlapping', kde=True, log_scale=plot_proxy_log_scale,  x_label=f'Proxy', y_label='Kind')
                log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_misclassification{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Saved {proxy_name} histogram (misclassification) to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_histograms_id_vs_ood{self.suffix}.pdf')))
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not misclassification plots for {proxy_name}. Reason {e}')
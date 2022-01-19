import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

from .base import *
import data.constants as dconstants
import evaluation.constants as econstants
from evaluation.util import run_model_on_datasets, get_data_loader
from evaluation.logging import *
import evaluation.callbacks
from plot.util import plot_histograms
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

    def _get_distribution_labels_perturbations(self, **kwargs):
        """ Gets labels for id vs ood where ood data is left out classes.
        
        Returns:
        --------
        auroc_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for auroc calculation.
        auroc_mask : torch.Tensor, shape [N]
            Which samples should be used for AUROC calculation.
        distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions.
        distribution_label_names : dict
            Mapping that names all the labels in `distribution_labels`.
        """

        is_ood = run_model_on_datasets(None, 
            [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on],
            run_model = False,
            callbacks = [
                evaluation.callbacks.make_callback_count_neighbours_with_attribute(
                    lambda data, outputs: data.is_out_of_distribution.numpy(), 0
                )
            ]
        )[0]
        distribution_labels = (torch.cat(is_ood, dim = 0) > 0).long()
        auroc_mask = torch.ones_like(distribution_labels).bool()
        auroc_labels = torch.zeros_like(distribution_labels).bool()
        auroc_labels[distribution_labels == econstants.ID_CLASS_NO_OOD_CLASS_NBS] = True
        distribution_label_names = {
            1 : f'Perturbed', 
            0 : f'Unperturbed', 
        }
        return auroc_labels, auroc_mask, distribution_labels, distribution_label_names

    def _get_distribution_labels_leave_out_classes(self, **kwargs):
        """ Gets labels for id vs ood where ood data is left out classes.
        
        Returns:
        --------
        auroc_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for auroc calculation.
        auroc_mask : torch.Tensor, shape [N]
            Which samples should be used for AUROC calculation.
        distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions.
        distribution_label_names : dict
            Mapping that names all the labels in `distribution_labels`.
        """
        # Classify vertices according to which distribution they belong to
        cfg: configuration.ExperimentConfiguration = kwargs['config']
        k = len(cfg.model.hidden_sizes) + 1
        distribution_labels = evaluation.util.get_distribution_labels(
            [get_data_loader(name, kwargs['data_loaders']) for name in self.evaluate_on],
            k, # Receptive field of each vertex is num-hidden-layers + 1
            mask = True,
            threshold = self.separate_distributions_tolerance,
        )
        auroc_labels, auroc_mask = evaluation.util.separate_distributions(distribution_labels, self.separate_distributions_by)
        distribution_label_names = {
            econstants.OOD_CLASS_NO_ID_CLASS_NBS : f'ood, no id nbs in {k} hops', 
            econstants.OOD_CLASS_ID_CLASS_NBS : f'ood, id nbs in {k}-hops',
            econstants.ID_CLASS_NO_OOD_CLASS_NBS : f'id, no ood nbs in {k}-hops', 
            econstants.ID_CLASS_ODD_CLASS_NBS : f'id, ood nbs in {k}-hops'
        }
        return auroc_labels, auroc_mask, distribution_labels, distribution_label_names

    def get_distribution_labels(self, **kwargs):
        """ Gets labels for id vs ood where ood data is left out classes.

        Parameters:
        -----------
        kind : str
            With which method to get out of distribution data.
        
        Returns:
        --------
        auroc_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for auroc calculation.
        auroc_mask : torch.Tensor, shape [N]
            Which samples should be used for AUROC calculation.
        distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions.
        distribution_label_names : dict
            Mapping that names all the labels in `distribution_labels`.
        """
        cfg: configuration.ExperimentConfiguration = kwargs['config']
        if cfg.data.ood_type == dconstants.LEFT_OUT_CLASSES:
            return self._get_distribution_labels_leave_out_classes(**kwargs)
        elif cfg.data.ood_type == dconstants.PERTURBATION:
            return self._get_distribution_labels_perturbations(**kwargs)
        else:
            raise RuntimeError(f'Could not separate distribution labels (id vs ood) by unknown type {cfg.data.ood_type}.')

class OODDetection(OODSeparation):
    """ Pipeline member to perform OOD detection for a given metric. Evaluates AUROC scores and logs plots. """

    name = 'OODDetection'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def ood_detection(self, proxy, labels, proxy_name, auroc_labels, auroc_mask, distribution_labels, distribution_label_names,
                        plot_proxy_log_scale=True,**kwargs):
        """ Performs ood detection and logs metrics and plots.
        
        Parameters:
        -----------
        proxy : torch.Tensor, shape [N]
            The proxy for separating id and ood. Higher values should be assigned to id data.
        labels : torch.Tensor, shape [N]
            Ground truth labels. Used to separate the proxy values by ground truth label.
        proxy_name : str
            Name of the proxy to use.
        auroc_labels : torch.Tensor, shape [N]
            Labels per sample assigning them to a certain distribution, used for auroc calculation.
        auroc_mask : torch.Tensor, shape [N]
            Which samples should be used for AUROC calculation.
        distribution_labels : torch.Tensor, shape [N]
            Labels for different types of distributions.
        distribution_label_names : dict
            Mapping that names all the labels in `distribution_labels`.
        plot_proxy_log_scale : bool
            If `True`, the proxy will be plotted in log scale.
        """

        if plot_proxy_log_scale:
            proxy += 1e-10 # To be able to plot

        # Calculate area under the ROC for separating in-distribution (label 1) from out of distribution (label 0)
        roc_auc = roc_auc_score(auroc_labels[auroc_mask].cpu().long().numpy(), proxy[auroc_mask].cpu().numpy()) # higher proxy -> higher uncertainty
        kwargs['metrics'][f'auroc_{proxy_name}{self.suffix}'] = roc_auc
        log_metrics(kwargs['logs'], {f'auroc_{proxy_name}{self.suffix}' : roc_auc}, f'{proxy_name}_plots')

        # Calculate the area under the PR curve
        precision, recall, _ = precision_recall_curve(auroc_labels[auroc_mask].cpu().long().numpy(), proxy[auroc_mask].cpu().numpy())
        aucpr = auc(recall, precision)
        kwargs['metrics'][f'aucpr_{proxy_name}{self.suffix}'] = aucpr
        log_metrics(kwargs['logs'], {f'aucpr_{proxy_name}{self.suffix}' : aucpr}, f'{proxy_name}_plots')

        try:
            # Log histograms and metrics label-wise
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
            pipeline_log(f'Could not create label-wise plots for {proxy_name}.')

        try:
            if self.log_plots:
                fig, ax = plot_histograms(proxy.cpu(), distribution_labels.cpu(), 
                    label_names=distribution_label_names,
                    kind='vertical', kde=True, log_scale=plot_proxy_log_scale,  x_label=f'Proxy', y_label='Kind')
                log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_all_kinds{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Saved {proxy_name} (all kinds) histogram to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_histograms_all_kinds{self.suffix}.pdf')))
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not distribution label-wise plots for {proxy_name}.')

        try:
            if self.log_plots:
                fig, ax = plot_histograms(proxy[auroc_mask].cpu(), auroc_labels[auroc_mask].cpu().long(), 
                    label_names={0 : 'Out ouf distribution', 1 : 'In distribution'},
                    kind='overlapping', kde=True, log_scale=plot_proxy_log_scale,  x_label=f'Proxy', y_label='Kind')
                log_figure(kwargs['logs'], fig, f'{proxy_name}_histograms_id_vs_ood{self.suffix}', f'{proxy_name}_plots', kwargs['artifacts'], save_artifact=kwargs['artifact_directory'])
                pipeline_log(f'Saved {proxy_name} histogram (id vs ood) to ' + str(osp.join(kwargs['artifact_directory'], f'{proxy_name}_histograms_id_vs_ood{self.suffix}.pdf')))
                plt.close(fig)
        except Exception as e:
            pipeline_log(f'Could not id vs ood plots for {proxy_name}. Reason {e}')

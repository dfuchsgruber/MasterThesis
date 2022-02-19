from torch import Tensor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging

from evaluation.logging import *

def misclassification_detection(proxy: Tensor, is_correctly_classified: Tensor, proxy_name: str, 
    plot_proxy_log_scale: bool=True, log_plots: bool=True, metric_suffix='', **kwargs):
    """ Performs missclassification detection and logs plots if so desired.

    Parameters:
    -----------
    proxy : torch.Tensor, shape [N]
        The proxy to use. Higher proxies should indicate more certainty w.r.t. to correct classifications.
    is_correctly_classified : torch.Tensor, shape [N]
        If a sample is correctly classified.
    proxy_name : str
        The name of the proxy
    plot_proxy_log_scale : bool, optional, default: True
        If the proxy will be plotted in log scale.
    log_plots : bool, optional, default: True
        If plots will be logged
    metric_suffix : str, optional, default: ''
        Suffix to append to all metrics.
    kwargs : dict
        Keyword arguments from the pipeline, containing e.g. the logger.
    """

    if plot_proxy_log_scale:
        proxy[proxy <= 0] = 1e-10

    roc_auc = roc_auc_score(is_correctly_classified.long().cpu().numpy(), proxy.cpu().numpy()) # higher proxy -> higher uncertainty
    kwargs['metrics'][f'misclassification_auroc_{proxy_name}{metric_suffix}'] = roc_auc
    log_metrics(kwargs['logs'], {f'misclassification_auroc_{proxy_name}{metric_suffix}' : roc_auc}, f'misclassification_{proxy_name}_plots')
    logging.info(f'misclassification_auroc_{proxy_name}{metric_suffix} : {roc_auc}')

    precision, recall, _ = precision_recall_curve(is_correctly_classified.long().cpu().numpy(), proxy.cpu().numpy())
    aucpr = auc(recall, precision)
    kwargs['metrics'][f'misclassification_aucpr_{proxy_name}{metric_suffix}'] = aucpr
    log_metrics(kwargs['logs'], {f'misclassification_aucpr_{proxy_name}{metric_suffix}' : aucpr}, f'misclassification_{proxy_name}_plots')

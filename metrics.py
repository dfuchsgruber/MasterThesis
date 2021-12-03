import numpy as np
from util import calibration_curve

def accuracy(logits, y_gnd):
    """ Computes the accuracy of the model's predictions. 
    
    Parameters:
    -----------
    logits : torch.Tensor, shape [batch_size, num_classes]
        Softmax logits for each class assignment predicted by the model.
    y_gnd : torch.Tensor, shape [batch_size]
        Ground-truth class label.
        
    Returns:
    --------
    accuracy : float
        Fraction of correctly predicted samples.
    """
    return (logits.argmax(1) == y_gnd).sum().item() / logits.shape[0]

def expected_calibration_error(probs, y_true, bins=30, eps=1e-12):
    """ Computes the expected calibration error as in [1].
    
    Parameters:
    -----------
    probs : torch.Tensor, shape [n, num_classes]
        Predicted probabilities.
    y_true : torch.Tensor, shape [n]
        True class labels.
    bins : int
        The number of bins to use.

    Returns:
    --------
    Expected calibration error.
    """
    _, bin_confidence, bin_accuracy, bin_weight = calibration_curve(probs, y_true, bins=bins, eps=eps)
    return (bin_weight * np.abs(bin_confidence - bin_accuracy)).sum()
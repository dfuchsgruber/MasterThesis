import numpy as np

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

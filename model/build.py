from typing import Any
import model.constants as mconst
from configuration import ExperimentConfiguration
from model.semi_supervised_node_classification import SemiSupervisedNodeClassification

def make_model(config: ExperimentConfiguration, num_inputs: int, num_outputs: int) -> Any:
    """ Instanciates a model from a configuration. The model must still be trained (or loaded from a checkpoint). 

    Parameters:
    -----------
    config : ExperimentConfiguration
        The experiment configuration to make a model from
    num_inputs : int
        The number of input features
    num_outputs : int
        The number of output features (i.e. number of classes)

    Returns:
    --------
    model : Any
        The model without any pre-loaded weights.
    """
    if config.model.model_type in ():
        pass
    else:
        # Fallback to standard semi supervised node classification pl wrapper
        model = SemiSupervisedNodeClassification(
            config.model, 
            num_inputs, 
            num_outputs, 
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    return model
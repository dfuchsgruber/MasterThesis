from typing import List
from torch_geometric.data import Data
import logging
import model.constants as mconst
import torch.nn as nn
import torch

class ParameterlessBase(nn.Module):
    """ Base class for a parameterless model that 'remembers' all training instances. """
    
    training_type = mconst.TRAIN_PARAMETERLESS

    def __init__(self):
        super().__init__()
        self._train_instances = None

    def clear_and_disable_cache(self):
        logging.info('Parameterless model base class disabled and cleared cache (with no effect).')

    def fit(self, batch: Data):
        """ Memorizes all instances in the train dataset using the `vertex_to_idx` attribute. 
        
        batch : Data
            The batch to memorize.
        """
        if not self.training:
            raise RuntimeError(f'Model was not set to eval mode.')
        self._train_instances = [v for v, idx in batch.vertex_to_idx.items() if batch.mask[idx]]
    
    def get_fit_idxs(self, batch: Data) -> List[int]:
        """ Gets the idx of train instances on any given graph. 
        
        Parameters:
        -----------
        batch : Data
            On which graph to get train instances.
        
        Returns:
        --------
        idxs : List[int]
            The train idxs.
        """
        if self._train_instances is None:
            raise RuntimeError(f'Model was not fit to any data.')
        idxs = []
        for v in self._train_instances:
            if v not in batch.vertex_to_idx:
                logging.warn(f'Instance {v} was in training data but not in new graph.')
            else:
                idx = batch.vertex_to_idx[v]
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                idxs.append(idx)
        return idxs

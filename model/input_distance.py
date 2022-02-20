from model.parameterless import ParameterlessBase
from model.prediction import *
from configuration import ModelConfiguration
from torch_geometric.data import Data
import torch

class InputDistance(ParameterlessBase):
    """ Simple baseline that uses the distance to input features as predictions and uncertainty proxy. """

    def __init__(self, config: ModelConfiguration, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.centroids = config.input_distance.centroids
        self.k = config.input_distance.k
        self.p = config.input_distance.p
        self.sigma = config.input_distance.sigma

    def __call__(self, batch: Data, *args, **kwargs) -> Prediction:
        idxs_train = self.get_fit_idxs(batch)

        x_train = batch.x[idxs_train] # num_train, D
        y_train = batch.y[idxs_train]
        x_eval = batch.x

        if self.centroids:
            raise NotImplemented
        else:
            distances = torch.cdist(x_eval, x_train, p=self.p) # [num_eval x num_train]
            evidence = torch.zeros(x_eval.size(0), self.num_classes)
            for y in range(self.num_classes):
                dist_y_sorted = torch.sort(distances[:, y_train == y], dim=-1)[0]
                if self.k > 0:
                    dist_y_sorted = dist_y_sorted[:, :self.k]
                evidence[:, y] = dist_y_sorted.mean(-1)

        # gaussian kernel
        evidence = torch.exp(-evidence / self.sigma)
        
        soft = evidence / (evidence.sum(1)[:, None] + 1e-10)
        hard = evidence.argmax(1)

        return Prediction(
            features=None,
            # The evidence can be used as proxy for uncertainty
            # We consider: i) evidence for the predicted class ii) evidence for all classes
            evidence_prediction = evidence[torch.arange(evidence.size(0)), hard],
            evidence_total = evidence.sum(1),
            **{
                SOFT_PREDICTIONS : soft,
                HARD_PREDICTIONS : hard,
            }
        )
        
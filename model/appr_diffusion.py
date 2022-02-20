from model.parameterless import ParameterlessBase
from model.prediction import *
from configuration import ModelConfiguration
from torch_geometric.data import Data
import torch_geometric.transforms as T
from util import approximate_page_rank_matrix
import torch

class APPRDiffusion(ParameterlessBase):
    """ Simple baseline that diffuses the train labels using approximate page rank. """

    def __init__(self, config: ModelConfiguration, num_classes: int):
        super().__init__()
        self.teleportation_probability = config.appnp.teleportation_probability
        self.diffusion_iterations = config.appnp.diffusion_iterations
        self.num_classes = num_classes
        self.self_loop_fill_value = config.self_loop_fill_value

    def __call__(self, batch: Data, *args, **kwargs) -> Prediction:
        batch = T.AddSelfLoops(fill_value=self.self_loop_fill_value)(batch)
        idxs_train = self.get_fit_idxs(batch)

        evidence = torch.zeros(batch.x.size(0), self.num_classes)
        for idx in idxs_train:
            evidence[idx, batch.y[idx]] = 1.0

        # Diffuse evidence
        ppr = torch.Tensor(approximate_page_rank_matrix(batch.edge_index.numpy(), batch.x.size(0),
            diffusion_iterations = self.diffusion_iterations, alpha = self.teleportation_probability))
        evidence = torch.matmul(ppr, evidence) # N x C

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
        
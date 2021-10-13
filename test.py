import torch

from train import train_model_semi_supervised_node_classification
from model.gnn import GNN
from data.linqs import Citeseer
from data.util import data_get_num_attributes, data_get_num_classes, stratified_split

data = Citeseer()
model = GNN('gcn', data_get_num_attributes(data), [16, 16], data_get_num_classes(data), use_spectral_norm=True)
if torch.cuda.is_available():
    model.cuda()

mask_train, mask_val, mask_test = torch.tensor(stratified_split(data[0].y.numpy(), 1, [0.05, 0.15, 0.8]))

train_model_semi_supervised_node_classification(model, data[0], mask_train[0], mask_val[0], epochs=5, early_stopping_patience=10)
import torch
import numpy as np

from model.train import train_model_semi_supervised_node_classification
from model.gnn import make_model_by_configuration
from model.semi_supervised_node_classification import SemiSupervisedNodeClassification
from data.gust_dataset import GustDataset
from data.util import data_get_num_attributes, data_get_num_classes, stratified_split_with_fixed_test_set_portion, SplitDataset
from seed import model_seeds
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl

data = GustDataset('cora_ml')
mask_split, mask_fixed = stratified_split_with_fixed_test_set_portion(data[0].y.numpy(), 5, portion_train=0.05, portion_val=0.15, portion_test_fixed=0.2, portion_test_not_fixed=0.6)
data_loader_train = DataLoader(SplitDataset(data, mask_split[0, 0]) , batch_size=1, shuffle=False)
data_loader_val = DataLoader(SplitDataset(data, mask_split[1, 0]) , batch_size=1, shuffle=False)

d = SplitDataset(data, mask_split[1, 0])
print(len(d))

# Build the model
cnfg = {
    'model_type' : 'appnp',
    'hidden_sizes' : [64],
    'use_bias' : True,
    'use_spectral_norm' : True,
    'weight_scale' : 5.0,
    'activation' : 'leaky_relu',
    'num_heads' : 2,
    'normalize' : True,
    'diffusion_iterations' : 2,
    'teleportation_probability' : 0.1,
}
backbone = make_model_by_configuration(cnfg, data_get_num_attributes(data[0]), data_get_num_classes(data[0]))
gnn = SemiSupervisedNodeClassification(backbone, learning_rate=1e-3)
trainer = pl.Trainer(max_epochs=50)
trainer.fit(gnn, data_loader_train, data_loader_val)
print(trainer.validate(gnn, data_loader_val))

# data = GustDataset('cora_ml')[0]
# mask_split, mask_fixed = stratified_split_with_fixed_test_set_portion(data.y.numpy(), 5, portion_train=0.05, portion_val=0.15, portion_test_fixed=0.2, portion_test_not_fixed=0.6)

# model_name = 'gcn'
# model_seeds = model_seeds(50, model_name=model_name)
# torch.manual_seed(model_seeds[0])

# model = GNN(model_name, data_get_num_attributes(data), [64], data_get_num_classes(data), use_spectral_norm=True, activation='leaky_relu', 
#     upper_lipschitz_bound=5, num_heads=8, diffusion_iterations=5)
# if torch.cuda.is_available():
#     model.cuda()

# train_history, val_history, best_model_state_dict, best_epoch = train_model_semi_supervised_node_classification(model, data, mask_split[0, 0], mask_split[1, 0], 
#     epochs=1000, early_stopping_patience=100, 
#     early_stopping_metrics={'val_loss' : 'min'}, learning_rate=1e-3)

# for metric, history in val_history.items():
#     print(f'Val {metric} of best model {history[best_epoch]}')
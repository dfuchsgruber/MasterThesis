import torch
import numpy as np

from model.train import train_model_semi_supervised_node_classification
from model.gnn import GNN
from data.gust_dataset import GustDataset
from data.util import data_get_num_attributes, data_get_num_classes, stratified_split_with_fixed_test_set_portion
from seed import model_seeds

data = GustDataset('cora_ml')[0]
mask_split, mask_fixed = stratified_split_with_fixed_test_set_portion(data.y.numpy(), 5, portion_train=0.05, portion_val=0.15, portion_test_fixed=0.2, portion_test_not_fixed=0.6)

model_name = 'gcn'
model_seeds = model_seeds(50, model_name=model_name)
torch.manual_seed(model_seeds[0])

model = GNN(model_name, data_get_num_attributes(data), [64], data_get_num_classes(data), use_spectral_norm=True, activation='leaky_relu', 
    upper_lipschitz_bound=5, num_heads=8, diffusion_iterations=5)
if torch.cuda.is_available():
    model.cuda()

train_history, val_history, best_model_state_dict, best_epoch = train_model_semi_supervised_node_classification(model, data, mask_split[0, 0], mask_split[1, 0], 
    epochs=1000, early_stopping_patience=100, 
    early_stopping_metrics={'val_loss' : 'min'}, learning_rate=1e-3)

for metric, history in val_history.items():
    print(f'Val {metric} of best model {history[best_epoch]}')
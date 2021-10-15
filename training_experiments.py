from sacred import Experiment
import numpy as np
import torch
import seml
from collections import defaultdict
import os.path as osp

from model.train import train_model_semi_supervised_node_classification
from model.gnn import GNN
from data.gust_dataset import GustDataset
from data.util import data_get_num_attributes, data_get_num_classes, stratified_split_with_fixed_test_set_portion
from seed import model_seeds

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

class ExperimentWrapper:

    def __init__(self, init_all=True, output_dir=None):
        if init_all:
            self.init_all()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @ex.capture(prefix="data")
    def init_dataset(self, dataset, num_dataset_splits, train_portion, val_portion, test_portion, test_portion_fixed):
        self.data = GustDataset(dataset)[0]
        self.data_mask_split, self.data_mask_test_fixed = stratified_split_with_fixed_test_set_portion(data.y.numpy(), num_dataset_splits, 
            portion_train=train_portion, portion_val=val_portion, portion_test_fixed=test_portion_fixed, portion_test_not_fixed=test_portion)

    @ex.capture(prefix="model")
    def init_model(self, model_type: str, hidden_sizes: list, weight_scale: float, num_initializations: int, use_spectral_norm: bool, num_heads=-1, 
        diffusion_iterations=5, teleportation_probability=0.1, use_bias=True, activation='leaky_relu', leaky_relu_slope=0.01):
        self.model_config = {
            'hidden_sizes' : hidden_sizes,
            'weight_scale' : weight_scale,
            'use_spectral_norm' : bool(use_spectral_norm),
            'num_heads' : num_heads,
            'diffusion_iterations' : diffusion_iterations,
            'teleportation_probability' : teleportation_probability,
            'type' : model_type,
            'use_bias' : bool(use_bias),
            'activation' : activation,
            'leaky_relu_slope' : leaky_relu_slope,
        }
        self.model_seeds = model_seeds(num_initializations, model_name=model_type)

    @ex.capture(prefix="optimization")
    def init_optimizer(self, learning_rate):
        self.optimization_config = {
            'learning_rate' : learning_rate
        }

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.init_optimizer()

    @ex.capture(prefix="training")
    def train(self, early_stopping_patience, num_epochs):
        best_results = defaultdict(list)
        for split_idx in range(self.data_mask_split.shape[1]):
            mask_train, mask_val = self.data_mask_split[0, split_idx], self.data_mask_split[1, split_idx]
            for reinitialization, seed in enumerate(self.model_seeds):
                torch.manual_seed(seed)

                # model = GNN(self.model_config['type'], data_get_num_attributes(self.data), self.model_config['hidden_sizes'], 
                #             data_get_num_classes(self.data), use_spectral_norm=self.model_config['use_spectral_norm'], activation='leaky_relu', 
                #             upper_lipschitz_bound=self.model_config['weight_scale'], num_heads=self.model_config['num_heads'], 
                #             diffusion_iterations=self.model_config['diffusion_iterations'], teleportation_probability=self.model_config['teleportation_probability'],
                #             use_bias=self.model_config['use_bias'], activation=self.model_config['activation'], leaky_relu_slope=self.model_config['leaky_relu_slope'],)
                
                # if torch.cuda.is_available():
                #     model.cuda()

                # train_history, val_history, best_model_state_dict, best_epoch = train_model_semi_supervised_node_classification(model, self.data, 
                #     mask_train, mask_val, epochs=num_epochs, early_stopping_patience=early_stopping_patience,  
                #     early_stopping_metrics={'val_loss' : 'min'}, learning_rate=self.learning_rate)

                # for metric, history in val_history.items():
                #     # print(f'Val {metric} of best model {history[best_epoch]}')
                #     results[metric].append(history[best_epoch])
                # result['best_epoch'].append(best_epoch)
                results[split_idx][reinitialization] = 1.0 # For testing
        with open(osp.join(self.output_dir, 'test_artifact.txt')) as f:
            f.write(f'{self.model_config}')
        return dict(results)


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(_config, experiment=None,):
    run_id = _config['overwrite']
    db_collection = _config['db_collection']
    if experiment is None:
        experiment = ExperimentWrapper(output_dir=os.path.join('..', 'artifacts', f'collection_{db_collection}_run_{run_id}'))
    return experiment.train()

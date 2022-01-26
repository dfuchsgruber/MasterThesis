
from collections.abc import Mapping
import data.constants as dconstants
import attr
from attrs import validators
from typing import List, Union, Optional, Any
from functools import wraps
import os.path as osp

def make_cls_converter(cls):
    """ Converter function that initializes a class with a dict or keeps the class as is. """
    def _convert(value):
        if isinstance(value, cls):
            return value
        elif isinstance(value, Mapping):
            return cls(**value)
        else:
            return ValueError
    return _convert

class BaseConfiguration:
    """ Base configuration class. """
    @property
    def registry_configuration(self):
        return attr.asdict(self, filter = lambda a, v: a.metadata.get('registry_attribute', True))

@attr.s
class ModelConfiguration(BaseConfiguration):
    """ Configuration for model initialization """ 
    hidden_sizes: List[int] = attr.ib(default=[64,], validator=lambda s, a, v: all(isinstance(x, int) for x in v))
    weight_scale: Optional[float] = attr.ib(default=1.0, validator=attr.validators.instance_of(float), converter=float)
    use_spectral_norm: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    model_type: str = attr.ib(default='gcn', validator=validators.in_(('gcn', 'gat', 'gin', 'sage', 'appnp')), converter=lambda s: s.lower())
    use_bias: bool =  attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    activation: bool = attr.ib(default='leaky_relu', validator=validators.in_(('leaky_relu', )), converter=lambda s: s.lower())
    leaky_relu_slope: bool = attr.ib(default=1e-2, validator=attr.validators.instance_of(float), converter=float)
    residual: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    freeze_residual_projection: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    dropout: float = attr.ib(default=0.0, validator=validators.and_(validators.ge(0), validators.le(1)), converter=float)
    drop_edge: float = attr.ib(default=0.0, validator=validators.and_(validators.ge(0), validators.le(1)), converter=float)
    use_spectral_norm_on_last_layer: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    cached: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    self_loop_fill_value: float = attr.ib(default=1.0, converter=float)

    # GAT configuration
    num_heads: int = attr.ib(default=8, validator=lambda s, a, v: isinstance(v, int) and v > 0, converter=int)

    # APPNP configuration
    diffusion_iterations: int = attr.ib(default=5, validator=lambda s, a, v: isinstance(v, int) and v > 0, converter=int)
    teleportation_probability: Optional[float] = attr.ib(default=0.2, validator=attr.validators.instance_of(float), converter=float)

    # GIN configuration
    normalize: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)

@attr.s
class DataConfiguration(BaseConfiguration):
    """ Configuration for dataset splitting and building. """
    dataset: str = attr.ib(default='cora_ml', validator=validators.in_(('cora_ml', 'cora_full', 'pubmed', 'citeseer',)), converter=lambda s: s.lower())
    train_portion: int = attr.ib(default=20, validator=validators.ge(0), converter=int)
    val_portion: int = attr.ib(default=20, validator=validators.ge(0), converter=int)
    test_portion_fixed: float = attr.ib(default=1, validator=validators.ge(0), converter=float)

    base_labels: Union[str, List[Any]] = attr.ib(default='all')
    train_labels: Union[str, List[Any]] = attr.ib(default='all')
    corpus_labels: Union[str, List[Any]] = attr.ib(default='all')
    left_out_class_labels: Union[str, List[Any]] = attr.ib(default=[])

    max_attempts_per_split: int = attr.ib(default=5, validator=validators.gt(0), converter=int)
    
    # How many id train vertices will be dropped from the ood graph in a hybrid setting
    drop_train_vertices_portion: float = attr.ib(default=0.1, validator=validators.and_(validators.ge(0), validators.le(1)), converter=float)

    # Transductive of hybrid setting
    setting: str = attr.ib(default=dconstants.TRANSDUCTIVE, validator=validators.in_((dconstants.TRANSDUCTIVE, dconstants.HYBRID)))
    ood_type: str = attr.ib(default=dconstants.PERTURBATION, validator=validators.in_((dconstants.LEFT_OUT_CLASSES, dconstants.PERTURBATION)))
    
    # For ood val and test datasets, should vertices be sampled from the available pool or should the entire pool be selected ?
    ood_sampling_strategy: str = attr.ib(default=dconstants.SAMPLE_ALL, validator=validators.in_((dconstants.SAMPLE_ALL, dconstants.SAMPLE_UNIFORM)))

    # How train / val / test splits are generated: Uniform samples `portion` vertices for each split uniformly per class
    split_type: str = attr.ib(default='uniform', validator=validators.in_(('uniform', )), converter=lambda s: s.lower())
    type: str = attr.ib(default='npz', validator=validators.in_(('npz',)), converter=lambda s: s.lower())

    # For perturbation experiments
    perturbation_budget: float = attr.ib(default=0.1, converter=float, validator=validators.and_(validators.ge(0), validators.le(1)))

    # Generate numerical features from text
    min_token_frequency: int = attr.ib(default=10, validator=validators.ge(0), converter=int)
    preprocessing: str = attr.ib(default='bag_of_words', validator=validators.in_(('bag_of_words', 'word_embedding')), converter=lambda s: s.lower())

    # If word embeddings are used
    language_model: str = attr.ib(default='bert-base-uncased')
    normalize: Optional[str] = attr.ib(default='l2', validator=validators.in_(('l1', 'l2', None)))
    vectorizer: str = attr.ib(default='tf-idf', validator=validators.in_(('tf-idf', 'count')), converter=lambda s: s.lower())

@attr.s
class EarlyStoppingConfiguration(BaseConfiguration):
    """ Configuration for early stopping """

    patience: int = attr.ib(default=100, validator=validators.ge(0), converter=int)
    mode: str = attr.ib(default='min', validator=validators.in_(('min', 'max')))
    monitor: str = attr.ib(default='val_loss')
    min_delta: float = attr.ib(default=1e-3, validator=validators.ge(0), converter=float)

@attr.s
class TrainingConfiguration(BaseConfiguration):
    """ Configuration for training a model. """
    max_epochs: int = attr.ib(default=1000, validator=validators.ge(0), converter=int)
    learning_rate: float = attr.ib(default=1e-3, validator=validators.ge(0), converter=float)
    early_stopping: EarlyStoppingConfiguration = attr.ib(default={}, converter=make_cls_converter(EarlyStoppingConfiguration))
    gpus: int = attr.ib(default=1, converter=int, metadata={'registry_attribute' : False})
    weight_decay: float = attr.ib(default=1e-3, converter=float, validator=validators.ge(0))
    suppress_stdout: bool = attr.ib(default=True, converter=bool, metadata={'registry_attribute' : False})
    train_model: bool = attr.ib(default=True, converter=bool, metadata={'registry_attribute' : False})
    self_training: bool = attr.ib(default=False, converter=bool)
    num_warmup_epochs: int = attr.ib(default=50, converter=int)

@attr.s
class EvaluationConfiguration(BaseConfiguration):
    """ Configuration for the pipeline """
    pipeline: List = attr.ib(default=[])
    print_pipeline: bool = attr.ib(default=True, converter=bool)
    ignore_exceptions: bool = attr.ib(default=False, converter=bool)
    log_plots: bool = attr.ib(default=False, converter=bool)
    save_artifacts: bool = attr.ib(default=False, converter=bool)

DEFAULT_REGISTRY_COLLECTION_NAME = 'model_registry'

@attr.s
class RunConfiguration(BaseConfiguration):
    """ Configuration for run names """
    name: str = attr.ib(default='', metadata={'registry_attribute' : False})
    args: List[str] = attr.ib(default=[], metadata={'registry_attribute' : False})

    num_initializations: int =  attr.ib(default=1, validator=validators.gt(0), converter=int, metadata={'registry_attribute' : False})
    num_dataset_splits: int = attr.ib(default=1, validator=validators.ge(0), converter=int, metadata={'registry_attribute' : False})

    model_registry_collection_name: str = attr.ib(default=DEFAULT_REGISTRY_COLLECTION_NAME, converter=str, metadata={'registry_attribute' : False})

@attr.s
class _RegistryConfiguration(BaseConfiguration):
    """ Configuration for the model registry. Values should never be initialized manually as they are not really configuration and more information. """
    split_idx: int = attr.ib(default=0, converter=int, validator=validators.ge(0))
    model_seed: int = attr.ib(default=0)

@attr.s
class EnsembleConfiguration(BaseConfiguration):
    """ Configuration for ensemble training. """ 
    num_members: int = attr.ib(default=1, validator=validators.gt(0), converter=int, metadata={'registry_attribute' : False})
    num_samples: int = attr.ib(default=1, validator=validators.gt(0), converter=int, metadata={'registry_attribute' : False})

@attr.s
class LoggingConfiguration(BaseConfiguration):
    """ Configuration for logging. """
    artifact_dir: str = attr.ib(default=osp.join('/', 'nfs', 'students', 'fuchsgru', 'artifacts'), metadata={'registry_attribute' : False})
    logging_dir: str = attr.ib(default=osp.join('/', 'nfs', 'students', 'fuchsgru', 'wandb'), metadata={'registry_attribute' : False})

@attr.s
class ExperimentConfiguration(BaseConfiguration):
    """ Configuration for a whole experiment """

    model: ModelConfiguration = attr.ib(default={}, converter=make_cls_converter(ModelConfiguration))
    data: DataConfiguration = attr.ib(default={}, converter=make_cls_converter(DataConfiguration))
    evaluation: EvaluationConfiguration = attr.ib(default={}, converter=make_cls_converter(EvaluationConfiguration), metadata={'registry_attribute' : False})
    training: TrainingConfiguration = attr.ib(default={}, converter=make_cls_converter(TrainingConfiguration))
    run: RunConfiguration = attr.ib(default={}, converter=make_cls_converter(RunConfiguration))
    ensemble: EnsembleConfiguration = attr.ib(default={}, converter=make_cls_converter(EnsembleConfiguration))
    registry: _RegistryConfiguration = attr.ib(default={}, converter=make_cls_converter(_RegistryConfiguration))
    logging: LoggingConfiguration = attr.ib(default={}, converter=make_cls_converter(LoggingConfiguration), metadata={'registry_attribute' : False})

if __name__ == '__main__':
    cfg_dict = ExperimentConfiguration().registry_configuration
    cfg = ExperimentConfiguration(**cfg_dict)
    print(cfg.registry_configuration == cfg_dict)
    print(cfg.registry_configuration)

from collections.abc import Mapping
from wsgiref.validate import validator
import data.constants as dconstants
import model.constants as mconstants
import attr
from attrs import validators
from typing import Dict, List, Union, Optional, Any
import os.path as osp
import yaml
import logging
from util import make_key_collatable

DEFAULT_REGISTRY_COLLECTION_NAME = 'model_registry_v5'
DEFAULT_DATASET_REGISTRY_COLLECTION_NAME = 'dataset_registry'

def make_cls_converter(cls, optional=False):
    """ Converter function that initializes a class with a dict or keeps the class as is. """
    def _convert(value):
        if optional and value is None:
            return None
        elif isinstance(value, cls):
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
class ReconstructionConfiguration(BaseConfiguration):
    """ Configuration for the reconstruction component in a model model. """
    loss_weight: float = attr.ib(default=0.0, converter=float) # If 0.0, no reconstruction will be applied
    sample: bool = attr.ib(default=True, converter=bool)
    num_samples: int = attr.ib(default=100, converter=int)
    seed: int = attr.ib(default=1337, converter=int)
    reconstruction_type: str = attr.ib(default=mconstants.AUTOENCODER, converter=str, validator=validators.in_((
        mconstants.AUTOENCODER, mconstants.TRIPLET, mconstants.ENERGY,
    ))) 
    cached: bool = attr.ib(default=True, converter=bool)
    margin_constrastive_loss: float = attr.ib(default=0.0, converter=float)

@attr.s
class FeatureReconstructionConfiguration(BaseConfiguration):
    """ Configuration for feature reconstruction in a model. """
    loss_weight: float = attr.ib(default=0.0, converter=float) # If 0.0, no reconstruction will be applied
    loss: str = attr.ib(default='l2', validator=attr.validators.in_(('l2', 'l1', 'bce', 'weighted_bce')))
    mirror_encoder: bool = attr.ib(default=True, converter=bool)
    activation_on_last_layer: bool = attr.ib(default=False, converter=bool)
    log_metrics_every: int = attr.ib(default=1, converter=int, validator=validators.gt(0), metadata={'registry_attribute' : False})
    num_samples : int = attr.ib(default=-1, converter=int)
    seed: int = attr.ib(default=1337, converter=int)

@attr.s
class GATConfiguration(BaseConfiguration):
    """ Configuration for GATs. """
    num_heads: int = attr.ib(default=8, validator=lambda s, a, v: isinstance(v, int) and v > 0, converter=int)

@attr.s
class BayesianGCNConfiguration(BaseConfiguration):
    """ Configuration for BGCNs. """
    sigma_1: float = attr.ib(default=1.0, validator=attr.validators.gt(0.0), converter=float)
    sigma_2: float = attr.ib(default=1e-6, validator=attr.validators.gt(0.0), converter=float)
    pi: float = attr.ib(default=0.75, validator=validators.and_(validators.ge(0), validators.le(1)))
    q_weight: float = attr.ib(default=1.0, converter=float)
    prior_weight = float = attr.ib(default=1.0, converter=float)

@attr.s
class APPNPConfiguration(BaseConfiguration):
    """ Configuration for APPNPs. """
    diffusion_iterations: int = attr.ib(default=10, validator=lambda s, a, v: isinstance(v, int) and v > 0, converter=int)
    teleportation_probability: Optional[float] = attr.ib(default=0.2, validator=attr.validators.instance_of(float), converter=float)

@attr.s
class InputDistanceConfiguration(BaseConfiguration):
    """ Configuration for the parameterless input distance baseline. """
    centroids: bool = attr.ib(default=False, converter=bool)
    # Averages the k closest distances within a class. If k < 0, all distances within a class are averaged 
    k: bool = attr.ib(default=-1, converter=int)
    p = attr.ib(default=2) # Norm to use for distance
    sigma: float = attr.ib(default=1.0, converter=float, validator=attr.validators.gt(0)) # scale parameter for the kernel

@attr.s
class GraphDirichletKernelConfiguration(BaseConfiguration):
    """ Configuration for the parameterless graph dirichlet kernel baseline. """
    sigma: float = attr.ib(default=1.0, converter=float, validator=attr.validators.gt(0)) # scale parameter for the kernel
    # How the kernel distances (evidences) are aggregated within a class
    reduction: str = attr.ib(default='sum', validator=attr.validators.in_(('sum', 'mul', 'mean', 'min', 'max')))

@attr.s
class LaplaceBayesianGCNConfiguration(BaseConfiguration):
    """ Configuration for laplace posterior approximation. """
    seed: int = attr.ib(default=1337, converter=int)
    hessian_structure = attr.ib(default=mconstants.DIAG_HESSIAN, converter=lambda s: s.lower(), validator=validators.in_((mconstants.DIAG_HESSIAN, mconstants.FULL_HESSIAN)))
    batch_size: int = attr.ib(default=-1, converter=int) # If -1, the whole dataset is used (no batching) to optimize prior parameters

    gpu: bool = attr.ib(default=False, converter=bool, metadata={'registry_attribute' : False})

@attr.s
class OrthongonalGCNConfiguration(BaseConfiguration):
    """ Configuration for laplace posterior approximation. """
    random_input_projection: bool = attr.ib(default=False, converter=bool)
    bjorck_orthonormalzation_n_iter: int = attr.ib(default=1, converter=int, validator=validators.gt(0))
    bjorck_orthonormalzation_rescaling: float = attr.ib(default=1.0, converter=float, validator=validators.gt(0))

@attr.s
class ModelConfiguration(BaseConfiguration):
    """ Configuration for model initialization """ 
    hidden_sizes: List[int] = attr.ib(default=[64,], validator=lambda s, a, v: all(isinstance(x, int) for x in v))
    linear_classification: bool = attr.ib(default=False, converter=bool)
    weight_scale: Optional[float] = attr.ib(default=1.0, validator=attr.validators.instance_of(float), converter=float)
    use_spectral_norm: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    model_type: str = attr.ib(default=mconstants.GCN, validator=validators.in_(mconstants.MODEL_TYPES), converter=lambda s: s.lower())
    use_bias: bool =  attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    activation: bool = attr.ib(default=mconstants.LEAKY_RELU, validator=validators.in_((mconstants.LEAKY_RELU, mconstants.RELU,)), converter=lambda s: s.lower())
    leaky_relu_slope: bool = attr.ib(default=1e-2, validator=attr.validators.instance_of(float), converter=float)
    residual: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    residual_pre_activation: bool = attr.ib(default=True, validator=attr.validators.instance_of(bool), converter=bool)
    freeze_residual_projection: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    dropout: float = attr.ib(default=0.0, validator=validators.and_(validators.ge(0), validators.le(1)), converter=float)
    drop_edge: float = attr.ib(default=0.0, validator=validators.and_(validators.ge(0), validators.le(1)), converter=float)
    use_spectral_norm_on_last_layer: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    use_bjorck_norm_on_last_layer: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    use_forbenius_norm_on_last_layer: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    use_rescaling_on_last_layer: bool = attr.ib(default=False, validator=attr.validators.instance_of(bool), converter=bool)
    use_residual_on_last_layer: bool = attr.ib(default=False, converter=bool)
    use_rescaling: bool = attr.ib(default=False, converter=bool)
    cached: bool = attr.ib(default=True, validator=attr.validators.instance_of(bool), converter=bool) # Cache will be cleared and disabled after training
    self_loop_fill_value: float = attr.ib(default=1.0, converter=float)
    use_forbenius_norm: bool = attr.ib(default=False, converter=bool)
    use_bjorck_norm: bool = attr.ib(default=False, converter=bool)
    forbenius_norm: float = attr.ib(default=1.0, converter=float, validator=validators.gt(0))
    initialization_scale: float = attr.ib(default=1.0, converter=float, validator=validators.gt(0))

    reconstruction: ReconstructionConfiguration = attr.ib(default={}, converter=make_cls_converter(ReconstructionConfiguration))
    feature_reconstruction: FeatureReconstructionConfiguration = attr.ib(default={}, converter=make_cls_converter(FeatureReconstructionConfiguration))

    gat: Optional[GATConfiguration] = attr.ib(default=None, converter=make_cls_converter(GATConfiguration, optional=True))
    appnp: Optional[APPNPConfiguration] = attr.ib(default=None, converter=make_cls_converter(APPNPConfiguration, optional=True))
    bgcn: Optional[BayesianGCNConfiguration] = attr.ib(default=None, converter=make_cls_converter(BayesianGCNConfiguration, optional=True))
    laplace: Optional[LaplaceBayesianGCNConfiguration] = attr.ib(default=None, converter=make_cls_converter(LaplaceBayesianGCNConfiguration, optional=True))
    orthogonal: Optional[OrthongonalGCNConfiguration] = attr.ib(default=None, converter=make_cls_converter(OrthongonalGCNConfiguration, optional=True))

    # Parameterless baselines
    input_distance: Optional[InputDistanceConfiguration] = attr.ib(default=None, converter=make_cls_converter(InputDistanceConfiguration, optional=True))
    gdk: Optional[GraphDirichletKernelConfiguration] = attr.ib(default=None, converter=make_cls_converter(GraphDirichletKernelConfiguration, optional=True))

    # Graph-Post-Net configuration
    # latent_size: int = attr.ib(default=16, converter=int)
    # use_batched_flow: bool = attr.ib(default=True, converter=bool)
    # num_radial: int = attr.ib(default=10, converter=int)
    # num_maf: int = attr.ib(default=0, converter=int)
    # num_gaussians: int = attr.ib(default=0, converter=int)
    # alpha_evidence_scale: str = attr.ib(default='latent-new', converter='str')

def sanitize_labels(labels):
    if not isinstance(str, labels):
        return [make_key_collatable(k) for k in labels]
    else:
        return labels
    

@attr.s
class DataConfiguration(BaseConfiguration):
    """ Configuration for dataset splitting and building. """
    dataset: str = attr.ib(default='cora_ml', validator=validators.in_(dconstants.DATASETS), converter=lambda s: s.lower())
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
    split_type: str = attr.ib(default='uniform', validator=validators.in_(('uniform', 'predefined')), converter=lambda s: s.lower())
    type: str = attr.ib(default='npz', validator=validators.in_(('npz',)), converter=lambda s: s.lower())

    # For perturbation experiments
    perturbation_budget: float = attr.ib(default=0.1, converter=float, validator=validators.and_(validators.ge(0), validators.le(1)))

    # Generate numerical features from text
    min_token_frequency: int = attr.ib(default=10, validator=validators.ge(0), converter=int)
    preprocessing: str = attr.ib(default='none', validator=validators.in_(('bag_of_words', 'word_embedding', 'none',)), converter=lambda s: s.lower())

    # If word embeddings are used
    language_model: str = attr.ib(default='bert-base-uncased')
    normalize: Optional[str] = attr.ib(default='l2', validator=validators.in_(('l1', 'l2', None)))
    vectorizer: str = attr.ib(default='tf-idf', validator=validators.in_(('tf-idf', 'count')), converter=lambda s: s.lower())

    # Scale to the input features
    feature_scale: float = attr.ib(default=1.0, converter=float)

    integrity_assertion: bool = attr.ib(default=False, converter=bool, metadata={'registry_attribute' : False})

    # If the k-hop neighbourhood should be precomputed for all the graphs
    precompute_k_hop_neighbourhood: int = attr.ib(default=2, converter=int)

    use_dataset_registry: bool = attr.ib(default=True, converter=bool, metadata={'registry_attribute' : False})
    dataset_registry_collection_name: str = attr.ib(default=DEFAULT_DATASET_REGISTRY_COLLECTION_NAME, metadata={'registry_attribute' : False})
    dataset_registry_directory: str = attr.ib(default='/nfs/students/fuchsgru/dataset_registry', metadata={'registry_attribute' : False})

@attr.s
class EarlyStoppingConfiguration(BaseConfiguration):
    """ Configuration for early stopping """

    patience: int = attr.ib(default=100, validator=validators.ge(0), converter=int)
    mode: str = attr.ib(default='min', validator=validators.in_(('min', 'max')))
    monitor: str = attr.ib(default='val_loss')
    min_delta: float = attr.ib(default=1e-3, validator=validators.ge(0), converter=float)

@attr.s
class FinetuningConfiguration(BaseConfiguration):
    """ Configuration for model finetuning. """
    max_epochs: Optional[int] = attr.ib(default=10)
    min_epochs: Optional[int] = attr.ib(default=1)
    enable: bool = attr.ib(default=False)

    reconstruction: Optional[ReconstructionConfiguration] = attr.ib(default=None, converter=make_cls_converter(ReconstructionConfiguration, optional=True))
    feature_reconstruction: Optional[FeatureReconstructionConfiguration] = attr.ib(default=None, converter=make_cls_converter(FeatureReconstructionConfiguration, optional=True))

    # Deprecated, only to compatbility with previous registry
    # Pass `None` to not enable this kind of finetuning.
    reconstruction_weight: Optional[float] = attr.ib(default=None) 
    feature_reconstruction_weight: Optional[float] = attr.ib(default=None)

@attr.s
class TemperatureScalingConfiguration(BaseConfiguration):
    """ Configuration for applying temperature scaling post-hoc. """
    criterion: str = attr.ib(default=mconstants.NLL, validator=attr.validators.in_(mconstants.TEMPERATURE_SCALING_OBJECTIVES))
    learning_rate: float = attr.ib(default=1e-2, validator=attr.validators.gt(0))
    max_epochs: int = attr.ib(default=50, validator=attr.validators.gt(0))

@attr.s
class TrainingConfiguration(BaseConfiguration):
    """ Configuration for training a model. """
    max_epochs: Optional[int] = attr.ib(default=1000)
    min_epochs: Optional[int] = attr.ib(default=None)
    learning_rate: float = attr.ib(default=1e-3, validator=validators.ge(0), converter=float)
    early_stopping: EarlyStoppingConfiguration = attr.ib(default={}, converter=make_cls_converter(EarlyStoppingConfiguration))
    gpus: int = attr.ib(default=1, converter=int, metadata={'registry_attribute' : False})
    weight_decay: float = attr.ib(default=1e-3, converter=float, validator=validators.ge(0))
    suppress_stdout: bool = attr.ib(default=True, converter=bool, metadata={'registry_attribute' : False})
    train_model: bool = attr.ib(default=True, converter=bool, metadata={'registry_attribute' : False})
    self_training: bool = attr.ib(default=False, converter=bool)
    num_warmup_epochs: int = attr.ib(default=50, converter=int)

    # Singular value bounding
    singular_value_bounding: bool = attr.ib(default=False, converter=bool)
    singular_value_bounding_eps: float = attr.ib(default=1e-2, converter=float, validator=validators.ge(0))

    # additional regularizers
    orthonormal_weight_regularization_strength: float = attr.ib(default=0.0, converter=float, validator=validators.ge(0))
    orthonormal_weight_scale: float = attr.ib(default=1.0, converter=float, validator=validators.gt(0))

    # Finetuning
    finetuning: FinetuningConfiguration = attr.ib(default={}, converter=make_cls_converter(FinetuningConfiguration))

    # Temperature scaling
    temperature_scaling: TemperatureScalingConfiguration = attr.ib(default=None, converter=make_cls_converter(TemperatureScalingConfiguration, optional=True))

@attr.s
class EvaluationConfiguration(BaseConfiguration):
    """ Configuration for the pipeline """
    pipeline: List = attr.ib(default=[])
    print_pipeline: bool = attr.ib(default=True, converter=bool)
    ignore_exceptions: bool = attr.ib(default=False, converter=bool)
    log_plots: bool = attr.ib(default=False, converter=bool)
    save_artifacts: bool = attr.ib(default=False, converter=bool)
    sample: bool = attr.ib(default=False, converter=bool)
    use_gpus: bool = attr.ib(default=False, converter=bool)


@attr.s
class RunConfiguration(BaseConfiguration):
    """ Configuration for run names """
    name: str = attr.ib(default='', metadata={'registry_attribute' : False})
    args: List[str] = attr.ib(default=[], metadata={'registry_attribute' : False})

    num_initializations: int =  attr.ib(default=1, validator=validators.gt(0), converter=int, metadata={'registry_attribute' : False})
    num_dataset_splits: int = attr.ib(default=1, validator=validators.gt(0), converter=int, metadata={'registry_attribute' : False})

    # Split and initialization idx are not used in the registry
    # Instead seeds are derived from them and put in the registry to identify pre-trained models (i.e. find their checkpoints)
    split_idx: int = attr.ib(default=0, validator=validators.ge(0), converter=int, metadata={'registry_attribute' : False})
    initialization_idx: int = attr.ib(default=0, validator=validators.ge(0), converter=int, metadata={'registry_attribute' : False})

    use_pretrained_model: bool = attr.ib(default=True, converter=bool, metadata={'registry_attribute' : False})
    model_registry_collection_name: str = attr.ib(default=DEFAULT_REGISTRY_COLLECTION_NAME, converter=str, metadata={'registry_attribute' : False})
    model_registry_directory: str = attr.ib(default='/nfs/students/fuchsgru/model_registry', metadata={'registry_attribute' : False})

    # If set, some values are updated with defaults depending on the dataset
    use_default_configuration: bool = attr.ib(default=False, converter=bool, metadata={'registry_attribute' : False})

@attr.s
class _RegistryConfiguration(BaseConfiguration):
    """ Configuration for the model registry. Values should never be initialized manually as they are not really configuration and more information. """
    split_seed: int = attr.ib(default=0)
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

    # Gradient logging
    log_gradients: Dict[str, str] = attr.ib(default={}, metadata={'registry_attribute' : False})
    log_gradients_relative_to_parameter: bool = attr.ib(default=True, converter=bool, metadata={'registry_attribute' : False})
    log_gradients_relative_to_norm: bool = attr.ib(default=True, converter=bool, metadata={'registry_attribute' : False})

    # Spectrum logging
    log_weight_matrix_spectrum_every: int = attr.ib(default=0, converter=int, metadata={'registry_attribute' : False})
    log_weight_matrix_spectrum_to_file: Optional[str] = attr.ib(default=None, metadata={'registry_attribute' : False})
    
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

def get_default_configuration_by_dataset(dataset: str) -> Dict:
    """ Gets the default values for a given dataset. 
    
    Parameters:
    -----------
    dataset : str
        The dataset to get the default configuration for.
    
    Returns:
    --------
    configuration : dict
        The default configuration values.
    """
    # dir = osp.dirname('__file__') # Doesn't work, because seml won't copy the configuration files to the server...
    dir = '.'
    cfg_fns = {
        dconstants.AMAZON_COMPUTERS : 'amazon_computers.yaml',
        dconstants.AMAZON_PHOTO : 'amazon_photo.yaml',
        dconstants.CITESEER : 'citeseer.yaml',
        dconstants.COAUTHOR_CS : 'coauthor_cs.yaml',
        dconstants.COAUTHOR_PHYSICS : 'coauthor_physics.yaml',
        dconstants.CORA_FULL : 'cora_full.yaml',
        dconstants.CORA_ML : 'cora_ml.yaml',
        dconstants.OGBN_ARXIV : 'ogbn_arxiv.yaml',
        dconstants.PUBMED : 'pubmed.yaml',
    }
    cfg_path = osp.join(dir, 'configs', cfg_fns[dataset])
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def update_with_default_configuration(config: ExperimentConfiguration):
    """ Updates the values in a configuration with default values. 
    
    Parameters:
    -----------
    config : ExperimentConfiguration
        The experiment configuration to update.
    """
    default_config = get_default_configuration_by_dataset(config.data.dataset)

    def recursive_update(d, config, prefix=[]):
        for k, v in config.items():
            if isinstance(v, Mapping):
                recursive_update(getattr(d, k), v, prefix=prefix + [k])
            else:
                setattr(d, k, v)
                logging.info(f'Set configuration value ' + '.'.join(prefix + [k]) + f' to default {v}')
    recursive_update(config, default_config)

    # In a perturbations setting, set base and corpus labels to train labels
    if config.data.ood_type == dconstants.PERTURBATION:
        config.data.base_labels = config.data.train_labels
        logging.info(f'Set configuration value data.base_labels to {config.data.train_labels}')
        config.data.corpus_labels = config.data.train_labels
        logging.info(f'Set configuration value data.corpus_labels to {config.data.corpus_labels}')
        config.data.left_out_class_labels = []
        logging.info(f'Set configuration value data.left_out_class_labels to {config.data.left_out_class_labels}')
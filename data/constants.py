
# TRAIN, VAL and TEST share the same graph and have disjunct masks
TRAIN = 'train' # Training graph with mask on train labels
VAL = 'val' # Training graph with mask on train labels
TEST = 'test' # Testing graph, all labels and full graph

# OOD has a different graph in the LoC setting that includes left out classes. OOD and OOD_TEST have disjunct masks
OOD_VAL = 'ood-val'
OOD_TEST = 'ood-test'

TRANSDUCTIVE = 'transductive'
HYBRID = 'hybrid'

LEFT_OUT_CLASSES = 'left-out-classes'
PERTURBATION = 'perturbations'

BERNOULLI = 'bernoulli'
NORMAL = 'normal'

# Different sampling strategies
SAMPLE_UNIFORM = 'uniform' # Sample uniformly per class
SAMPLE_ALL = 'all' # Don't sample but instead use all available samples in mask
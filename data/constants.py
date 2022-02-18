
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

# Datasets
CORA_FULL = 'cora_full'
CORA_ML = 'cora_ml'
DBLP = 'dblp'
PUBMED = 'pubmed'
CITESEER = 'citeseer'
OGBN_ARXIV = 'ogbn_arxiv'
COAUTHOR_CS = 'coauthor_cs'
COAUTHOR_PHYSICS = 'coauthor_physics'
AMAZON_COMPUTERS = 'amazon_computers'
AMAZON_PHOTO = 'amazon_photo'

DATASETS = (CORA_FULL, CORA_ML, DBLP, PUBMED, CITESEER, OGBN_ARXIV, COAUTHOR_PHYSICS, COAUTHOR_CS, AMAZON_PHOTO, AMAZON_COMPUTERS)
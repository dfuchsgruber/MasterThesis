# Model constants

# --- Model Types ---
GCN = 'gcn'
GAT = 'gat'
APPNP = 'appnp'
GIN = 'gin'
SAGE = 'sage'
BGCN = 'bgcn'
GCN_LINEAR_CLASSIFICATION = 'gcn_linear_classification'
GCN_LAPLACE = 'gcn_laplace'

# Baselines
APPR_DIFFUSION = 'appr_diffusion'
INPUT_DISTANCE = 'input_distance'
GDK = 'graph_dirichlet_kernel'

# Collect all valid model types
MODEL_TYPES = (GCN, APPNP, GAT, GIN, SAGE, BGCN, APPR_DIFFUSION, INPUT_DISTANCE, GDK, GCN_LINEAR_CLASSIFICATION, GCN_LAPLACE)

# --- Laplace Parameters ---
FULL_HESSIAN = 'full'
DIAG_HESSIAN = 'diag'


# --- Reconstruction types ---
AUTOENCODER = 'autoencoder'
TRIPLET = 'triplet'
ENERGY = 'energy'

# --- Loss types --
LEAKY_RELU = 'leaky_relu'
RELU = 'relu'

# --- Wrapper Types ---
TRAIN_PL = 'train-pytorch-lightning'
TRAIN_PARAMETERLESS = 'train-parameterless'
TRAIN_LAPLACE = 'train-laplace'
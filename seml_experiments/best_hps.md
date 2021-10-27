cora_ml-gcn:
  fixed:
    data.dataset: cora_ml
    model.model_type: gcn
    model.hidden_sizes: [64]
    training.learning_rate: 0.003

cora_ml-gat:
  fixed:
    data.dataset: cora_ml
    model.model_type: gat
    training.learning_rate: 0.003
    model.hidden_sizes: [64, 64]
    model.num_heads: 8


cora_ml-gin:
  fixed:
    data.dataset: cora_ml
    model.model_type: gin
    training.learning_rate: 0.003
    model.hidden_sizes: [64]


cora_ml-sage:
  fixed:
    data.dataset: cora_ml
    model.model_type: sage
    training.learning_rate: 0.003
    model.hidden_sizes: [64]
    model.normalize: True


cora_ml-mlp:
  fixed:
    data.dataset: cora_ml
    model.model_type: mlp
    training.learning_rate: 0.001
    model.hidden_sizes: [32]


cora_ml-appnp:
  fixed:
    data.dataset: cora_ml
    model.model_type: appnp
    training.learning_rate: 0.003
    model.hidden_sizes: [64, 64]
    model.teleportation_probability: 0.1
    model.diffusion_iterations: 10


citeseer-gat:
  fixed:
    data.dataset: citeseer
    model.model_type: gat
    training.learning_rate: 0.001
    model.hidden_sizes: [64, 64]
    model.num_heads: 8


citeseer-gcn:
  fixed:
    data.dataset: citeseer
    model.model_type: gcn
    training.learning_rate: 0.010
    model.hidden_sizes: [64]


citeseer-gin:
  fixed:
    data.dataset: citeseer
    model.model_type: gin
    training.learning_rate: 0.010
    model.hidden_sizes: [64]


citeseer-sage:
  fixed:
    data.dataset: citeseer
    model.model_type: sage
    training.learning_rate: 0.010
    model.hidden_sizes: [64]
    model.normalize: True


citeseer-mlp:
  fixed:
    data.dataset: citeseer
    model.model_type: mlp
    training.learning_rate: 0.010
    model.hidden_sizes: [64]


citeseer-appnp:
  fixed:
    data.dataset: citeseer
    model.model_type: appnp
    training.learning_rate: 0.003
    model.hidden_sizes: [32]
    model.teleportation_probability: 0.1
    model.diffusion_iterations: 10


pubmed-gat:
  fixed:
    data.dataset: pubmed
    model.model_type: gat
    training.learning_rate: 0.010
    model.hidden_sizes: [64, 64]
    model.num_heads: 8


pubmed-gcn:
  fixed:
    data.dataset: pubmed
    model.model_type: gcn
    training.learning_rate: 0.003
    model.hidden_sizes: [64, 64]


pubmed-gin:
  fixed:
    data.dataset: pubmed
    model.model_type: gin
    training.learning_rate: 0.003
    model.hidden_sizes: [64]


pubmed-sage:
  fixed:
    data.dataset: pubmed
    model.model_type: sage
    training.learning_rate: 0.010
    model.hidden_sizes: [64]
    model.normalize: True


pubmed-mlp:
  fixed:
    data.dataset: pubmed
    model.model_type: mlp
    training.learning_rate: 0.003
    model.hidden_sizes: [64, 64]


pubmed-appnp:
  fixed:
    data.dataset: pubmed
    model.model_type: appnp
    training.learning_rate: 0.003
    model.hidden_sizes: [64, 64]
    model.teleportation_probability: 0.1
    model.diffusion_iterations: 2

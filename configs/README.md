### Standard configurations

These yaml files list standard configurations for experiments on different datasets. Note that except in the case of `cora_full`, left out classes will be removed from the training labels in the LoC experiment. To be more precise, to treat the files correctly, apply the following in a corresponding setting:
- Leave out classes: Remove `left_out_class_labels` from `train_labels` from the configuration
- Perturbations: Set `left_out_class_labels` to `[]` (and leave `train_labels` as is)

Also standard architectural backbone choices (hidden sizes) are supplied per dataset.
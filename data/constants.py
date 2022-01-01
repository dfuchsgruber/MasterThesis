
# TRAIN, VAL and TEST share the same graph and have disjunct masks
TRAIN = 'train' # Training graph with mask on train labels
VAL = 'val' # Training graph with mask on train labels
TEST = 'test' # Testing graph, all labels and full graph

# OOD has a different graph in the LoC setting that includes left out classes. OOD and OOD_TEST have disjunct masks
OOD = 'ood'
OOD_TEST = 'ood-test'

TRANSDUCTIVE = ('transductive',)
HYBRID = ('hybrid',)

LEFT_OUT_CLASSES = ('left-out-classes', 'loc', 'left_out_classes')
PERTURBATION = ('perturbations', 'perturbation', 'noise')

BERNOULLI = ('bernoulli',)
NORMAL = ('normal',)
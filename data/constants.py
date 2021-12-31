
# TRAIN, VAL and TEST share the same graph and have disjunct masks
TRAIN = 'train' # Training graph with mask on train labels
VAL = 'val' # Training graph with mask on train labels
TEST = 'test' # Testing graph, all labels and full graph

# OOD has a different graph in the LoC setting that includes left out classes. OOD and OOD_TEST have disjunct masks
OOD = 'ood'
OOD_TEST = 'ood-test'

BASE = 'base' # Graph that all other graphs are based on


TRAIN_DROPPED = 'train-full' # Training graph before dropping vertices, mask on all vertices that were dropped
VAL_REDUCED = 'val-reduced' # Validation graph with mask on train labels, same graph as TRAIN
VAL_TRAIN_LABELS = 'val-train-labels' # Validation graph with mask on train labels, same graph as VAL
TEST_REDUCED = 'test-reduced' # Testing graph with mask on train labels, same graph as TRAIN
TEST_TRAIN_LABELS = 'test-train-labels' # Testing graph with mask on train labels, same graph as TEST



TRANSDUCTIVE = ('transductive',)
HYBRID = ('hybrid',)

LEFT_OUT_CLASSES = ('left-out-classes', 'loc', 'left_out_classes')
PERTURBATION = ('perturbations', 'perturbation', 'noise')

BERNOULLI = ('bernoulli',)
NORMAL = ('normal',)
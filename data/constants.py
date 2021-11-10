
TRAIN = 'train' # Training graph with mask on train labels, potentially edges and vertices were removed
VAL = 'val' # Validation graph with mask on validation labels, potentially edges and vertices were removed
VAL_REDUCED = 'val-reduced' # Validation graph with mask on train labels, same graph as TRAIN
VAL_TRAIN_LABELS = 'val-train-labels' # Validation graph with mask on train labels, same graph as VAL
TEST = 'test' # Testing graph, all labels and full graph
TEST_REDUCED = 'test-reduced' # Testing graph with mask on train labels, same graph as TRAIN
TEST_TRAIN_LABELS = 'test-train-labels' # Testing graph with mask on train labels, same graph as TEST
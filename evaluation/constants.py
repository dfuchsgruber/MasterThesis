import matplotlib.colors as mcolors
import numpy as np

# For LoC out-of-distribution detection and perturbations
ID_CLASS_NO_OOD_CLASS_NBS = 0
OOD_CLASS_NO_ID_CLASS_NBS = 1
# These are exclusively used in LoC
ID_CLASS_ODD_CLASS_NBS = 2
OOD_CLASS_ID_CLASS_NBS = 3

ID_CLASS = (ID_CLASS_NO_OOD_CLASS_NBS, ID_CLASS_ODD_CLASS_NBS,)
OOD_CLASS = (OOD_CLASS_NO_ID_CLASS_NBS, OOD_CLASS_ID_CLASS_NBS,)

# For plotting to make color-consistent plots
DISTRIBUTION_COLORS = {
    ID_CLASS_NO_OOD_CLASS_NBS : 'tab:orange',
    ID_CLASS_ODD_CLASS_NBS : tuple((np.array(mcolors.to_rgba('tab:orange')) + 0.15).clip(0, 1.0)),
    OOD_CLASS_NO_ID_CLASS_NBS : 'tab:blue',
    OOD_CLASS_ID_CLASS_NBS : tuple((np.array(mcolors.to_rgba('tab:blue')) + 0.15).clip(0, 1.0)),
}

COLOR_FIT = 'tab:green'
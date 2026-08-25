#--------------------------------------------------
#VALIDITY CHECKS
#--------------------------------------------------
from .wrappers_utils.validity import _validate_size, _class_probabilities

#--------------------------------------------------
#CENTRALIZATION
#--------------------------------------------------
from .wrappers_utils.measures import gini_index

#--------------------------------------------------
#DISTANCE BETWEEN DISTRIBUTIONS
#--------------------------------------------------
from .wrappers_utils.measures import gaussian_kl_divergence, gaussian_hellinger_distance

#--------------------------------------------------
#PLOTTING
#--------------------------------------------------
from .wrappers_utils.plotting import rescale, plot_heatmap, plot_elbow

#--------------------------------------------------
#SPECTRAL MATRICES
#--------------------------------------------------
from .wrappers_utils.matrices import adjacency_matrix, laplacian_matrix, random_walk_matrix

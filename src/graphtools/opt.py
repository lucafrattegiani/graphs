#--------------------------------------------------
#GRAPH DISTANCES
#--------------------------------------------------
from ._opt.distance import fused_gromov_wasserstein_distance

#--------------------------------------------------
#GRAPH COUPLING
#--------------------------------------------------
from ._opt.coupling import hungarian_coupling

#--------------------------------------------------
#PLOTTING
#--------------------------------------------------
from .plotting import plot_coupling_fgw
from .plotting import plot_coupling_hungarian
from .plotting import plot_heatmap

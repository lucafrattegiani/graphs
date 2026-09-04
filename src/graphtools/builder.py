#Mathematics and ML
import torch

#Data structures
from torch_geometric.data import Data

#--------------------------------------------------
#GRAPH CONSTRUCTION
#--------------------------------------------------
from .building.edges import voronoi

def build_graph(positions: torch.Tensor, method: str = "voronoi",
                add_voronoi: bool = True) -> Data:
    """Build a graph from two-dimensional spatial positions.

    Parameters
    ----------
    positions : torch.Tensor
        Two-dimensional spatial coordinates with shape ``[num_nodes, 2]``.
        The graph and its edges are stored on the same device as this tensor.
    method : str, default = "voronoi"
        Method used to construct the edges. It must be one of {"voronoi"}.
    add_voronoi : bool, default = True
        Whether to store the complete ``scipy.spatial.Voronoi`` object in the
        graph's ``cells`` attribute when it is available.

    Returns
    -------
    torch_geometric.data.Data
        Graph containing ``edge_index``, ``num_nodes``, and ``pos``. When
        ``add_voronoi`` is True, it also contains ``cells``.
    """
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    if not isinstance(add_voronoi, bool):
        raise TypeError("add_voronoi must be a boolean")

    edge_builders = {
        "voronoi": voronoi,
    }
    if method not in edge_builders:
        raise ValueError(
            f"Unsupported method {method!r}. "
            f"Choose one of: {', '.join(edge_builders)}."
        )

    edge_index, cells = edge_builders[method](
        positions,
        device = positions.device,
    )

    graph = Data(
        edge_index = edge_index,
        num_nodes = positions.shape[0],
        pos = positions,
    )
    if add_voronoi:
        graph.cells = cells

    return graph

#--------------------------------------------------
#PLOTTING
#--------------------------------------------------
from .plotting import plot_positions, plot_voronoi, plot_nodes, plot_graph

#Mathematics and ML
import torch

#Data structures
from torch_geometric.data import Data

#--------------------------------------------------
#GRAPH CONSTRUCTION
#--------------------------------------------------
from .building.edges import epsilon_voronoi, voronoi

def build_graph(positions: torch.Tensor, method: str = "voronoi",
                add_voronoi: bool = True,
                method_kwargs: dict | None = None) -> Data:
    """Build a graph from two-dimensional spatial positions.

    Parameters
    ----------
    positions : torch.Tensor
        Two-dimensional spatial coordinates with shape ``[num_nodes, 2]``.
        The graph and its edges are stored on the same device as this tensor.
    method : str, default = "voronoi"
        Method used to construct the edges. It must be one of
        {"voronoi", "epsilon_voronoi"}.
    add_voronoi : bool, default = True
        Whether to store the generated tessellation in the graph's ``cells``
        attribute. A standard Voronoi object is stored for ``"voronoi"``;
        a list of Shapely polygons is stored for ``"epsilon_voronoi"``.
    method_kwargs : dict | None, default = None
        Additional arguments passed to the selected construction method. For
        ``"epsilon_voronoi"``, these can include ``epsilon``, ``tolerance``,
        and ``resolution``.

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
    if method_kwargs is None:
        method_kwargs = {}
    elif not isinstance(method_kwargs, dict):
        raise TypeError("method_kwargs must be a dictionary or None")
    else:
        method_kwargs = method_kwargs.copy()
    if "device" in method_kwargs:
        raise ValueError("device is determined from positions and cannot be overridden")

    edge_builders = {
        "voronoi": voronoi,
        "epsilon_voronoi": epsilon_voronoi,
    }
    if method not in edge_builders:
        raise ValueError(
            f"Unsupported method {method!r}. "
            f"Choose one of: {', '.join(edge_builders)}."
        )

    edge_index, cells = edge_builders[method](
        positions,
        device = positions.device,
        **method_kwargs,
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
from .plotting import plot_epsilon_voronoi, plot_positions, plot_voronoi, plot_nodes, plot_graph

#Mathematics and ML
import torch
from torch_geometric.data import Data

#Plotting
import matplotlib.pyplot as plt

#--------------------------------------------------
#GRAPH SAMPLING
#--------------------------------------------------
from .wrappers_graphon.edges_sampler import ring_graphon, sbm_graphon, bipartite_graphon, random_graphon, sample_edges
from .wrappers_graphon.positions_sampler import polygonal, ellipse, square, circle, bubbles, ring, two_columns, sample_positions
from .wrappers_graphon.state_sampler import hard_state, soft_state, continuous_state, sample_states


def sample_graph(n: int, device: torch.device | str = "cpu",
                 structure: str = "random",
                 state: str | None = None,
                 positions: str | None = None,
                 structure_kwargs: dict | None = None,
                 state_kwargs: dict | None = None,
                 position_kwargs: dict | None = None) -> Data:
    """
    Sample a PyTorch Geometric graph from a graphon distribution with optional node states and positions.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    device : torch.device | str, default = "cpu"
        Device on which to store edges, states, and positions.
    structure : str, default = "random"
        Graphon used to sample the edges. It can be ``"ring"``, ``"random"``,
        ``"sbm"``, or ``"bipartite"``.
    state : str | None, default = None
        Optional node-state generator. It can be ``"hard"``, ``"soft"``,
        ``"continuous"``, or None to omit node states.
    positions : str | None, default = None
        Optional node-position generator. It can be ``"ring"``, ``"bubbles"``,
        ``"square"``, ``"circle"``, ``"two_columns"``, or None to omit
        positions.
    structure_kwargs : dict | None, default = None
        Additional arguments passed to the structure-specific sampling function.
    state_kwargs : dict | None, default = None
        Additional arguments passed to ``sample_states``.
    position_kwargs : dict | None, default = None
        Additional arguments passed to ``sample_positions``.

    Returns
    -------
    torch_geometric.data.Data
        Sampled graph containing ``edge_index`` and ``num_nodes``. The ``x``
        and ``pos`` attributes are included when ``state`` and ``positions``
        are specified, respectively.
    """
    structure_kwargs = {} if structure_kwargs is None else structure_kwargs
    state_kwargs = {} if state_kwargs is None else state_kwargs
    position_kwargs = {} if position_kwargs is None else position_kwargs

    if not isinstance(structure_kwargs, dict):
        raise TypeError("structure_kwargs must be a dictionary or None")
    if not isinstance(state_kwargs, dict):
        raise TypeError("state_kwargs must be a dictionary or None")
    if not isinstance(position_kwargs, dict):
        raise TypeError("position_kwargs must be a dictionary or None")

    edge_index = sample_edges(
        n = n,
        structure = structure,
        **structure_kwargs,
    ).to(device)

    graph = Data(
        edge_index = edge_index,
        num_nodes = n,
    )

    if state is not None:
        graph.x = sample_states(
            n = n,
            state_type = state,
            device = device,
            **state_kwargs,
        )

    if positions is not None:
        position_parameters = position_kwargs.copy()
        if positions == "bubbles":
            position_parameters["graph"] = graph

        graph.pos = sample_positions(
            n = n,
            position_type = positions,
            device = device,
            **position_parameters,
        )

    return graph


#--------------------------------------------------
#PLOTTING
#--------------------------------------------------
from .wrappers_graphon.positions_sampler import plot_positions
from .utils import adjacency_matrix, plot_heatmap


def plot_graph(graph: Data, device: torch.device | str = "cpu", title: str = "",
               figsize: tuple[float, float] = (15, 7), labels: list[str] | tuple[str, ...] | None = None) -> None:
    """
    Plot a graph's adjacency matrix and spatial representation side by side.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        Graph whose adjacency matrix and spatial representation are plotted.
    device : torch.device | str, default = "cpu"
        Device on which to compute the adjacency matrix.
    title : str, default = ""
        Overall figure title. If empty, the number of nodes and edges is used.
    figsize : tuple[float, float], default = (15, 7)
        Figure width and height in inches.
    Returns
    -------
    None
        The function displays a Matplotlib figure and does not return a value.

    Raises
    ------
    TypeError
        If ``graph`` is not a ``torch_geometric.data.Data`` object.
    """
    if not isinstance(graph, Data):
        raise TypeError("graph must be a torch_geometric.data.Data object")

    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        num_edges = 0
    elif graph.is_directed():
        num_edges = edge_index.size(1)
    else:
        num_edges = int((edge_index[0] < edge_index[1]).sum().item())

    adjacency = adjacency_matrix(
        graph,
        normalized = False,
        device = device,
        sparse = False,
    )

    fig, axes = plt.subplots(1, 2, figsize = figsize)
    plot_heatmap(
        adjacency,
        title = " ",
        ax = axes[0],
    )
    plot_positions(
        graph,
        title = " ",
        ax = axes[1],
        labels = labels,
    )

    overall_title = title or f"n={graph.num_nodes}, edges={num_edges}"
    fig.suptitle(overall_title)
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    plt.show()

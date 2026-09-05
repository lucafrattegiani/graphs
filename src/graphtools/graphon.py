#Mathematics and ML
import torch
from torch_geometric.data import Data

#--------------------------------------------------
#GRAPH SAMPLING
#--------------------------------------------------
from .sampling.edges import ring_graphon, sbm_graphon, bipartite_graphon, random_graphon, sample_edges
from .sampling.positions import polygonal, ellipse, square, circle, bubbles, ring, two_columns, sample_positions
from .sampling.states import hard_state, soft_state, continuous_state, sample_states

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
        device = device,
        **structure_kwargs,
    )

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
from .plotting import plot_nodes, plot_graph

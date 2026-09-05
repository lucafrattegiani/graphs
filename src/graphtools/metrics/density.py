#Torch data and computations
import torch
from torch_geometric.data import Data
from ..utils.validity import _graph_device

def density(graph: Data, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Computes the density of a graph.

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.

    Returns:
    -------
    torch.Tensor
        Density of the graph.
    """
    device = _graph_device(graph, device)
    if graph.num_nodes < 2: #Nan value for 1 node graphs
        return torch.tensor(torch.nan, dtype = torch.float32, device = device)
    value = graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1))
    return torch.tensor(value, dtype = torch.float32, device = device)

#Torch data and computations
import torch
from torch_geometric.data import Data

def density(graph: Data, device: torch.device | str = "cpu") -> torch.Tensor:
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
    if graph.num_nodes < 2: #Nan values for 1 node graphs
        return torch.full((graph.num_nodes,), torch.nan, dtype = torch.float32, device = device)
    density = graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1))
    return density
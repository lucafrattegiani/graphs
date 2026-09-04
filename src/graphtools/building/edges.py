#Mathematics and ML
import torch
from scipy.spatial import Voronoi
from torch_geometric.utils import to_undirected

def voronoi(positions: torch.Tensor, device: torch.device | str = "cpu") -> list[torch.Tensor | Voronoi | None]:
    """Create Voronoi edges and return the associated tessellation.

    Parameters
    ----------
    positions : torch.Tensor
        Two-dimensional spatial coordinates with shape ``[num_nodes, 2]``.
    device : torch.device | str, default = "cpu"
        Device on which to return the edge tensor.

    Returns
    -------
    list
        Two-element list containing the undirected edge indices with shape
        ``[2, num_edges]`` and the complete ``scipy.spatial.Voronoi`` object.
        For two or three points, all points are connected and the second
        element is None because no Voronoi tessellation is computed.
    """
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [num_nodes, 2]")
    num_nodes = positions.shape[0]
    if num_nodes < 2:
        raise ValueError("positions must be at least two points")

    if num_nodes <= 3: #For 2 or 3 points, just connect all of them
        voronoi = None
        edges = torch.triu_indices(
            num_nodes,
            num_nodes,
            offset = 1,
            device = device,
        ).t()
    else:
        voronoi = Voronoi(positions.detach().cpu().numpy()) #Compute Voronoi tessellation
        edges = torch.as_tensor(
            voronoi.ridge_points.copy(), #Extract pairs of nodes having adjacent Voronoi cells
            dtype = torch.long,
            device = device,
        )

    #Double each undirected edge
    edges = torch.sort(edges, dim = 1).values
    edges = torch.unique(edges, dim = 0)
    edges = edges.t().contiguous()
    edges = to_undirected(edges, num_nodes = num_nodes)

    return [edges, voronoi]
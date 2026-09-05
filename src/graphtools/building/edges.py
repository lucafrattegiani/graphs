#Mathematics and ML
import torch
import numpy as np

#Voronoi builder
from scipy.spatial import Voronoi

#Voronoi-epsilon builder
import shapely
from shapely.geometry import MultiPoint, Point, box
from shapely.strtree import STRtree
from shapely.geometry.polygon import Polygon

#Graph tools
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

def epsilon_voronoi(positions: torch.Tensor, device: torch.device | str = "cpu", 
                    epsilon: float = 0.3,
                    tolerance: float = 1e-9,
                    resolution: int = 32) -> list[torch.Tensor | list[Polygon] | None]:
    """Create epsilon-Voronoi edges and return the associated tessellation.

    Parameters
    ----------
    positions : torch.Tensor
        Two-dimensional spatial coordinates with shape ``[num_nodes, 2]``.
    epsilon : float
        Radius of the circle used to filter the Voronoi edges.
    tolerance : float
        Tolerance level for considering segments.
    resolution : int, default = 32
        Number of segments to approximate each circular arc.
    device : torch.device | str, default = "cpu"
        Device on which to return the edge tensor.

    Returns
    -------
    list
        Two-element list containing the undirected edge indices with shape
        ``[2, num_edges]`` and a list of the Voronoi cells as ``shapely.geometry.Polygon`` objects.
    """
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [num_nodes, 2]")
    num_nodes = positions.shape[0]
    if num_nodes < 2:
        raise ValueError("positions must be at least two points")

    # Convert positions to numpy array for compatibility
    positions = positions.detach().cpu().numpy().astype(np.float64)

    # Create the envelope: external square containing all points
    margin = 1.1 * epsilon #Margin to define square size
    min_xy = positions.min(axis = 0) - margin #Compute minimum across x coordinate and y coordinate and subtract the margin
    max_xy = positions.max(axis = 0) + margin #Compute minimum across x coordinate and y coordinate and subtract the margin

    # Box object represents the square
    envelope = box(
        min_xy[0], min_xy[1], #Bottom left corner
        max_xy[0], max_xy[1], #Top right corner
    )

    # Encode positions in shapely
    points = MultiPoint(positions)

    # Generate shapely Voronoi tessellation
    tessellation = shapely.voronoi_polygons(points, extend_to = envelope, ordered = True)
    vor_cells = tessellation.geoms #Extract iterable object of shapely Polygons

    # Intersect Voronoi cells with epsilon-balls
    vor_eps_cells = []
    for node, cell in enumerate(vor_cells):
        # Ball around each node
        circle = Point(positions[node]).buffer(epsilon, quad_segs = resolution)

        # Compute intersection 
        vor_eps_cell = cell.intersection(circle)
        vor_eps_cells.append(vor_eps_cell)

    # Encoding cells in spatial indexes STRtree
    spatial_index = STRtree(vor_eps_cells)

    # Construct edges between nodes whose Voronoi-epsilon cells touch
    edge_pairs = []

    for node, cell in enumerate(vor_eps_cells):
        # Find adjacent cells
        touching_nodes = spatial_index.query(
            cell,
            predicate = "dwithin",
            distance = tolerance,
        )

        # Add edges 
        for touching_node in touching_nodes:
            touching_node = int(touching_node)
            if touching_node <= node: #Avoid double counting edges and self-loops
                continue

            edge_pairs.append((node, touching_node))

    # Adapt edge list to pytorch geometric format
    edge_index = torch.tensor(edge_pairs, dtype=torch.long, device = device).t().contiguous()
    if edge_index.shape[0] == 0: #If there are no edges, create an empty tensor with shape [2, 0]
        edge_index = torch.empty((2, 0), dtype=torch.long, device = device)
    else:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes) #Ensure the edges are undirected
    
    return [edge_index, vor_eps_cells]

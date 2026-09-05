#Torch data and computations
import torch
from torch_geometric.data import Data
from .validity import _graph_device

def adjacency_matrix(graph: Data, normalized: bool = True, device: torch.device | str | None = None,
                     sparse: bool = True) -> torch.Tensor:
    """
    Computes the adjacency matrix of a graph:

    A_ij = w_{ij}

    w_{ij} = Edge weight between nodes i and j (1 if unweighted)

    If normalized, it is defined as:

    A_norm = D^(-1/2) A D^(-1/2)

    A = Adjacency matrix
    D = Diagonal degree matrix

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    normalized : bool
        Whether to symmetrically normalize the adjacency matrix.
    device : torch.device | str
        Device to perform computations on.
    sparse : bool
        Whether to return a sparse tensor. If False, returns a dense tensor.

    Returns:
    -------
    torch.Tensor
        Dense or sparse COO adjacency matrix of shape (num_nodes, num_nodes).
    """
    device = _graph_device(graph, device)
    num_nodes = graph.num_nodes
    if num_nodes < 2:  # Nan values for graphs with less than 2 nodes
        raise ValueError("Graph must have at least 2 nodes")

    # No edges case
    if graph.edge_index is None:
        edge_index = torch.empty((2, 0), dtype = torch.long, device = device)
    else:
        edge_index = graph.edge_index.to(device)

    if not sparse:
        # Initialize dense adjacency matrix
        A = torch.zeros((num_nodes, num_nodes), dtype = torch.float32, device = device)
        A[edge_index[0], edge_index[1]] = 1.0

        # Degree normalization
        if normalized:
            degrees = A.sum(dim = 1)
            inv_sqrt_degrees = torch.zeros_like(degrees)
            non_isolated = degrees > 0
            inv_sqrt_degrees[non_isolated] = degrees[non_isolated].rsqrt()
            A = inv_sqrt_degrees[:, None] * A * inv_sqrt_degrees[None, :]

        adjacency = A

    else:
        # Initialize sparse adjacency matrix and merge repeated edges
        values = torch.ones(edge_index.size(1), dtype = torch.float32, device = device)
        with torch.sparse.check_sparse_tensor_invariants(enable = True):
            adjacency = torch.sparse_coo_tensor(
                edge_index,
                values,
                size = (num_nodes, num_nodes),
                device = device,
            ).coalesce()
        indices = adjacency.indices()
        values = torch.ones(indices.size(1), dtype = torch.float32, device = device)

        # Degree normalization
        if normalized:
            degrees = torch.zeros(num_nodes, dtype = values.dtype, device = device)
            degrees.index_add_(0, indices[0], values)
            inv_sqrt_degrees = torch.zeros_like(degrees)
            non_isolated = degrees > 0
            inv_sqrt_degrees[non_isolated] = degrees[non_isolated].rsqrt()
            values = values * inv_sqrt_degrees[indices[0]] * inv_sqrt_degrees[indices[1]]

        with torch.sparse.check_sparse_tensor_invariants(enable = True):
            adjacency = torch.sparse_coo_tensor(
                indices,
                values,
                size = (num_nodes, num_nodes),
                device = device,
                is_coalesced = True,
            )

    return adjacency

def laplacian_matrix(graph: Data, normalized: bool = True, device: torch.device | str | None = None,
                     sparse: bool = True) -> torch.Tensor:
    """
    Computes the Laplacian matrix of a graph. Defined as:

    L = D - A -> Non normalized
    L_norm = D^(-1/2) L D^(-1/2) -> Normalized

    A = Adjacency matrix
    D = Diagonal degree matrix

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    normalized : bool
        Whether to symmetrically normalize the Laplacian matrix.
    device : torch.device | str
        Device to perform computations on.
    sparse : bool
        Whether to return a sparse tensor. If False, returns a dense tensor.

    Returns:
    -------
    torch.Tensor
        Dense or sparse COO Laplacian matrix of shape (num_nodes, num_nodes).
    """
    device = _graph_device(graph, device)
    adjacency = adjacency_matrix(graph, normalized = False, device = device, sparse = sparse)
    num_nodes = graph.num_nodes

    if not sparse:
        degrees = adjacency.sum(dim = 1)

        if normalized:
            inv_sqrt_degrees = torch.zeros_like(degrees)
            non_isolated = degrees > 0
            inv_sqrt_degrees[non_isolated] = degrees[non_isolated].rsqrt()
            laplacian = -inv_sqrt_degrees[:, None] * adjacency * inv_sqrt_degrees[None, :]
            diagonal = torch.arange(num_nodes, device = device)
            laplacian[diagonal, diagonal] += non_isolated.to(laplacian.dtype)
        else:
            laplacian = torch.diag(degrees) - adjacency

    else:
        indices = adjacency.indices()
        adjacency_values = adjacency.values()
        degrees = torch.zeros(num_nodes, dtype = adjacency_values.dtype, device = device)
        degrees.index_add_(0, indices[0], adjacency_values)
        non_isolated_nodes = torch.nonzero(degrees > 0, as_tuple = False).flatten()
        diagonal_indices = torch.stack((non_isolated_nodes, non_isolated_nodes))

        if normalized:
            inv_sqrt_degrees = torch.zeros_like(degrees)
            inv_sqrt_degrees[non_isolated_nodes] = degrees[non_isolated_nodes].rsqrt()
            edge_values = (
                -adjacency_values
                * inv_sqrt_degrees[indices[0]]
                * inv_sqrt_degrees[indices[1]]
            )
            diagonal_values = torch.ones(non_isolated_nodes.numel(), dtype = adjacency_values.dtype, device = device)
        else:
            edge_values = -adjacency_values
            diagonal_values = degrees[non_isolated_nodes]

        laplacian_indices = torch.cat((indices, diagonal_indices), dim = 1)
        laplacian_values = torch.cat((edge_values, diagonal_values))

        with torch.sparse.check_sparse_tensor_invariants(enable = True):
            laplacian = torch.sparse_coo_tensor(
                laplacian_indices,
                laplacian_values,
                size = (num_nodes, num_nodes),
                device = device,
            ).coalesce()

    return laplacian

def random_walk_matrix(graph: Data, device: torch.device | str | None = None,
                       sparse: bool = True, lazy: bool = True) -> torch.Tensor:
    """
    Computes the random walk transition matrix of a graph. Defined as:

    P = D^(-1) A
    P_lazy = 1/2 (I_n + P)

    A = Adjacency matrix
    D = Diagonal degree matrix
    I_n = Identity matrix

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    sparse : bool
        Whether to return a sparse tensor. If False, returns a dense tensor.
    lazy : bool
        Whether to return the lazy random walk matrix.

    Returns:
    -------
    torch.Tensor
        Dense or sparse COO random walk matrix of shape (num_nodes, num_nodes).
        In the non-lazy case, rows associated with isolated nodes contain only
        zeros.
    """
    device = _graph_device(graph, device)
    adjacency = adjacency_matrix(graph, normalized = False, device = device, sparse = sparse)
    num_nodes = graph.num_nodes

    if not sparse:
        degrees = adjacency.sum(dim = 1)
        inv_degrees = torch.zeros_like(degrees)
        non_isolated = degrees > 0
        inv_degrees[non_isolated] = degrees[non_isolated].reciprocal()
        random_walk = inv_degrees[:, None] * adjacency

        if lazy:
            identity = torch.eye(num_nodes, dtype = random_walk.dtype, device = device)
            random_walk = 0.5 * (identity + random_walk)

    else:
        indices = adjacency.indices()
        adjacency_values = adjacency.values()
        degrees = torch.zeros(num_nodes, dtype = adjacency_values.dtype, device = device)
        degrees.index_add_(0, indices[0], adjacency_values)
        values = adjacency_values / degrees[indices[0]]

        if lazy:
            diagonal = torch.arange(num_nodes, device = device)
            identity_indices = torch.stack((diagonal, diagonal))
            indices = torch.cat((indices, identity_indices), dim = 1)
            identity_values = torch.ones(num_nodes, dtype = values.dtype, device = device)
            values = 0.5 * torch.cat((values, identity_values))

            with torch.sparse.check_sparse_tensor_invariants(enable = True):
                random_walk = torch.sparse_coo_tensor(
                    indices,
                    values,
                    size = (num_nodes, num_nodes),
                    device = device,
                ).coalesce()
        else:
            with torch.sparse.check_sparse_tensor_invariants(enable = True):
                random_walk = torch.sparse_coo_tensor(
                    indices,
                    values,
                    size = (num_nodes, num_nodes),
                    device = device,
                    is_coalesced = True,
                )

    return random_walk

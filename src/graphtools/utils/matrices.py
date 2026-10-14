#Torch data and computations
import torch
from torch_geometric.data import Data
import networkx as nx
from ..validity import _graph_device

def inverse_shortest_path_matrix(graph: Data, device: torch.device | str | None = None,
                          sparse: bool = True) -> torch.Tensor:
    """
    Computes structural similarity between nodes according to shortest path distances.

    C_ij = 1 / (1 + d_ij) for connected pairs of nodes.
    C_ij = 0 for disconnected pairs of nodes.
    C_ii = 1 

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data. Edge directions and weights are ignored.
    device : torch.device | str | None
        Output device. If None, follow edge_index (CPU if absent).
        Computation uses local CPU data; the input graph is not modified.
    sparse : bool
        Whether to return a sparse tensor. If False, returns a dense tensor.

    Returns:
    -------
    torch.Tensor
        Float32 dense or sparse COO matrix of shape (num_nodes, num_nodes).
        The sparse output is coalesced and stores only connected pairs,
        including the diagonal.
    """
    output_device = _graph_device(graph, device)
    device = torch.device("cpu")
    num_nodes = graph.num_nodes
    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(num_nodes))
    if graph.edge_index is not None:
        graph_nx.add_edges_from(graph.edge_index.detach().cpu().t().tolist())

    if sparse:
        rows, columns, values = [], [], []
    else:
        matrix = torch.zeros((num_nodes, num_nodes), dtype = torch.float32, device = device)

    for source, distances in nx.all_pairs_shortest_path_length(graph_nx):
        targets = list(distances)
        inverse_distances = [1.0 / (1 + distances[target]) for target in targets]
        if sparse:
            rows.extend([source] * len(targets))
            columns.extend(targets)
            values.extend(inverse_distances)
        else:
            matrix[source, targets] = torch.tensor(inverse_distances, dtype = torch.float32, device = device)

    if sparse:
        indices = torch.tensor([rows, columns], dtype = torch.long, device = device)
        values = torch.tensor(values, dtype = torch.float32, device = device)
        with torch.sparse.check_sparse_tensor_invariants(enable = True):
            matrix = torch.sparse_coo_tensor(
                indices,
                values,
                size = (num_nodes, num_nodes),
                device = device,
            ).coalesce()

    return matrix.to(output_device)

def jaccard_similarity_matrix(graph: Data, device: torch.device | str | None = None,
                              sparse: bool = True) -> torch.Tensor:
    """
    Computes the Jaccard similarity between node neighborhoods.

    C_ij = |N(i) intersection N(j)| / |N(i) union N(j)|

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data. Edge directions, weights, duplicates and self-loops
        do not affect the neighborhoods.
    device : torch.device | str | None
        Output device. If None, follow edge_index (CPU if absent).
        Computation uses local CPU data; the input graph is not modified.
    sparse : bool
        Whether to return a sparse tensor. If False, returns a dense tensor.

    Returns:
    -------
    torch.Tensor
        Float32 dense or coalesced sparse COO matrix of shape
        (num_nodes, num_nodes). Sparse output stores only nonzero entries.
    """
    output_device = _graph_device(graph, device)
    device = torch.device("cpu")
    num_nodes = graph.num_nodes
    neighbors = [set() for _ in range(num_nodes)]
    if graph.edge_index is not None:
        for source, target in graph.edge_index.detach().cpu().t().tolist():
            if source != target:
                neighbors[source].add(target)
                neighbors[target].add(source)
    degrees = [len(neighborhood) for neighborhood in neighbors]

    if sparse:
        rows, columns, values = [], [], []
    else:
        matrix = torch.zeros((num_nodes, num_nodes), dtype = torch.float32, device = device)

    for source, neighborhood in enumerate(neighbors):
        # Count common neighbors only for pairs with a nonempty intersection.
        common_counts = {}
        for neighbor in neighborhood:
            for target in neighbors[neighbor]:
                common_counts[target] = common_counts.get(target, 0) + 1
        targets = list(common_counts)
        similarities = [
            common_counts[target] / (degrees[source] + degrees[target] - common_counts[target])
            for target in targets
        ]
        if sparse:
            rows.extend([source] * len(targets))
            columns.extend(targets)
            values.extend(similarities)
        elif targets:
            matrix[source, targets] = torch.tensor(similarities, dtype = torch.float32, device = device)

    if sparse:
        indices = torch.tensor([rows, columns], dtype = torch.long, device = device)
        values = torch.tensor(values, dtype = torch.float32, device = device)
        with torch.sparse.check_sparse_tensor_invariants(enable = True):
            matrix = torch.sparse_coo_tensor(
                indices,
                values,
                size = (num_nodes, num_nodes),
                device = device,
            ).coalesce()

    return matrix.to(output_device)

def adjacency_matrix(graph: Data, normalized: bool = False, device: torch.device | str | None = None,
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

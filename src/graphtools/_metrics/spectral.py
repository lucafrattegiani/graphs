#Torch data and computations
import torch
from torch_geometric.data import Data
from torch_kmeans import KMeans

from ..utils.matrices import adjacency_matrix, laplacian_matrix, random_walk_matrix
from ..validity import _graph_device

def spectral_decomposition(graph: Data, matrix: str = "adjacency", normalized: bool = True,
                           device: torch.device | str | None = None,
                           lazy: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the complete spectral decomposition of a specified matrix extracted from an undirected graph.

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Undirected graph data.
    matrix : str
        Matrix to decompose {"adjacency", "laplacian", "random_walk"}.
    normalized : bool
        Whether to symmetrically normalize the adjacency or Laplacian matrix
        before its spectral decomposition.
    device : torch.device | str | None, default = None
        Device on which to perform computations. If None, infer it from the graph.
    lazy : bool
        Whether to use the lazy random walk matrix. Used only when matrix is
        "random walk".

    Returns:
    -------
    tuple[torch.Tensor, torch.Tensor]
        All eigenvalues in descending order and the corresponding eigenvectors
        stored column-wise.
    """
    device = _graph_device(graph, device)
    if matrix == "adjacency":
        graph_matrix = adjacency_matrix(
            graph,
            normalized = normalized,
            device = device,
            sparse = False,
        )
    elif matrix == "laplacian":
        graph_matrix = laplacian_matrix(
            graph,
            normalized = normalized,
            device = device,
            sparse = False,
        )
    elif matrix == "random_walk":
        graph_matrix = random_walk_matrix(
            graph,
            device = device,
            sparse = False,
            lazy = lazy,
        )
    else:
        raise ValueError('matrix must be one of {"adjacency", "laplacian", "random_walk"}')

    if matrix == "random_walk":
        eigenvalues, eigenvectors = torch.linalg.eig(graph_matrix)
        # General eigendecomposition does not guarantee sorted eigenvalues.
        order = torch.argsort(eigenvalues.real, descending = True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(graph_matrix)
        # eigh already returns ascending eigenvalues: reverse the paired columns.
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

    return eigenvalues, eigenvectors

def spectral_embedding(graph: Data, K: int, device: torch.device | str | None = None, normalized: bool = True) -> torch.Tensor:
    """
    Computes a K-dimensional spectral embedding of an undirected graph from the
    leading eigenvectors of its normalized adjacency matrix. Defined as:

    S = D^(-1/2) A D^(-1/2)
    X = D^(-1/2) U_K

    A = Adjacency matrix
    D = Diagonal degree matrix
    U_K = Matrix containing the K leading eigenvectors of S

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Undirected graph data.
    K : int
        Number of spectral components.
    device : torch.device | str | None, default = None
        Device on which to perform computations. If None, infer it from the graph.
    normalized : bool
        Whether to normalize spectral coordinates.

    Returns:
    -------
    torch.Tensor
        Degree-normalized spectral coordinates with shape (num_nodes, K), one
        row for each graph node.
    """
    device = _graph_device(graph, device)
    if not isinstance(K, int) or isinstance(K, bool) or K < 1 or K > graph.num_nodes:
        raise ValueError("K must be an integer between 1 and graph.num_nodes")

    _, eigenvectors = spectral_decomposition(
        graph,
        matrix = "adjacency",
        normalized = True,
        device = device,
    )

    adjacency = adjacency_matrix(
        graph,
        normalized = False,
        device = device,
        sparse = False,
    )
    degrees = adjacency.sum(dim = 1)
    inv_sqrt_degrees = torch.zeros_like(degrees)
    non_isolated = degrees > 0
    inv_sqrt_degrees[non_isolated] = degrees[non_isolated].rsqrt()

    embedding = inv_sqrt_degrees[:, None] * eigenvectors[:, :K]

    if normalized:
        embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)

    return embedding


def spectral_clustering(graph: Data, device: torch.device | str | None = None,
                        K: int | None = None) -> torch.Tensor:
    """
    Cluster non-isolated nodes using normalized-adjacency spectral embeddings.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        Undirected graph. Isolated nodes are excluded before spectral analysis.
    device : torch.device | str | None, default = None
        Computation and output device. None follows the graph's edge indices.
        The input graph is never modified.
    K : int | None, default = None
        Number of clusters among non-isolated nodes; special labels do not count.
        None selects K from the largest eigengap of the non-isolated subgraph.
        An explicit K must be positive and not exceed its number of nodes.
        If numerical exclusions leave fewer than K usable nodes, K is reduced
        to their count. For an entirely isolated graph, a positive K is ignored.

    Returns
    -------
    torch.Tensor
        Integer labels in original node order, on the requested device.
        If all nodes are isolated, labels are 0, ..., num_nodes - 1.
        Otherwise isolated nodes receive -1. Non-isolated nodes with zero-norm
        or non-finite embedding rows receive -2 and are excluded from K-means.
        Finite positive norms are normalized without an arbitrary cutoff.
        A failed eigensolver or non-finite spectrum during automatic K selection
        assigns -2 to all non-isolated nodes. Other runtime/input errors propagate.
        Remaining nodes receive non-negative cluster labels.
    """
    if not isinstance(graph, Data):
        raise TypeError("graph must be a torch_geometric.data.Data object")
    device = _graph_device(graph, device)
    if K is not None and (not isinstance(K, int) or isinstance(K, bool) or K < 1):
        raise ValueError("K must be a positive integer or None")

    edges = graph.edge_index
    if edges is None:
        edges = torch.empty((2, 0), dtype = torch.long, device = device)
    else:
        edges = edges.to(device)
    active = torch.zeros(graph.num_nodes, dtype = torch.bool, device = device)
    active[edges.reshape(-1)] = True
    nodes = active.nonzero(as_tuple = True)[0]
    num_active = nodes.numel()
    if num_active == 0:
        return torch.arange(graph.num_nodes, dtype = torch.long, device = device)
    if K is not None and K > num_active:
        raise ValueError("K cannot exceed the number of non-isolated nodes")

    labels = torch.full((graph.num_nodes,), -1, dtype = torch.long, device = device)
    labels[nodes] = -2
    if num_active == 1:
        labels[nodes] = 0
        return labels

    # Remap only edge indices into a new graph; keep the original graph untouched.
    mapping = torch.full((graph.num_nodes,), -1, dtype = torch.long, device = device)
    mapping[nodes] = torch.arange(num_active, device = device)
    subgraph = Data(num_nodes = num_active, edge_index = mapping[edges])

    try:
        if K is None:
            eigenvalues, _ = spectral_decomposition(
                subgraph, matrix = "adjacency", normalized = True, device = device,
            )
            if not torch.isfinite(eigenvalues).all():
                return labels
            eigengaps = eigenvalues[:-1].real - eigenvalues[1:].real
            K = int(torch.argmax(eigengaps).item()) + 1

        # Inspect raw rows before normalization: do not turn NaN/Inf into data.
        coordinates = spectral_embedding(subgraph, K = K, device = device,
                                          normalized = False)
    except torch.linalg.LinAlgError:
        # No reliable node embedding is available if the eigensolver fails.
        return labels

    norms = torch.linalg.vector_norm(coordinates, dim = 1)
    usable = torch.isfinite(coordinates).all(dim = 1) & torch.isfinite(norms) & (norms > 0)
    usable_nodes = nodes[usable]
    num_usable = usable_nodes.numel()
    if num_usable == 0:
        return labels
    coordinates = coordinates[usable] / norms[usable, None]
    K = min(K, num_usable)

    # Handle trivial partitions directly, including the single remaining node.
    if K == 1:
        labels[usable_nodes] = 0
    elif K == num_usable:
        labels[usable_nodes] = torch.arange(num_usable, device = device)
    else:
        kmeans = KMeans(
            n_clusters = K,
            init_method = "k-means++",
            num_init = min(8, num_usable - 1),
            verbose = False,
        )
        result = kmeans(coordinates.unsqueeze(0))
        labels[usable_nodes] = result.labels.squeeze(0).to(device = device, dtype = torch.long)
    return labels

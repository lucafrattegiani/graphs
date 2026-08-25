#Torch data and computations
import torch
from torch_geometric.data import Data
from torch_kmeans import KMeans

from ..utils import adjacency_matrix, laplacian_matrix, random_walk_matrix

def spectral_decomposition(graph: Data, matrix: str = "adjacency", normalized: bool = True,
                           device: torch.device | str = "cpu",
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
    device : torch.device | str
        Device to perform computations on.
    lazy : bool
        Whether to use the lazy random walk matrix. Used only when matrix is
        "random walk".

    Returns:
    -------
    tuple[torch.Tensor, torch.Tensor]
        All eigenvalues in descending order and the corresponding eigenvectors
        stored column-wise.
    """
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
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(graph_matrix)

    order = torch.argsort(eigenvalues.real, descending = True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors

def spectral_embedding(graph: Data, K: int, device: torch.device | str = "cpu", normalized: bool = True) -> torch.Tensor:
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
    device : torch.device | str
        Device to perform computations on.
    normalized : bool
        Whether to normalize spectral coordinates.

    Returns:
    -------
    torch.Tensor
        Degree-normalized spectral coordinates with shape (num_nodes, K), one
        row for each graph node.
    """
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


def spectral_clustering(graph: Data, device: torch.device | str = "cpu",
                        K: int | None = None) -> torch.Tensor:
    """
    Cluster graph nodes using normalized-adjacency spectral embeddings.

    When ``K`` is not specified, the number of clusters is selected from the
    largest gap between consecutive eigenvalues of the normalized adjacency
    matrix. Clusters in spectral space are then assigned using K-means.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        Undirected graph to cluster.
    device : torch.device | str, default = "cpu"
        Device on which to perform the spectral decomposition and K-means.
    K : int | None, default = None
        Number of clusters. If None, determine it automatically from the
        maximum eigengap of the normalized adjacency matrix.

    Returns
    -------
    torch.Tensor
        Integer cluster assignments with shape ``[graph.num_nodes]``.

    """
    if not isinstance(graph, Data):
        raise TypeError("graph must be a torch_geometric.data.Data object")

    if K is None:
        eigenvalues, _ = spectral_decomposition(
            graph,
            matrix = "adjacency",
            normalized = True,
            device = device,
        )
        eigengaps = eigenvalues[:-1].real - eigenvalues[1:].real
        K = int(torch.argmax(eigengaps).item()) + 1

    if not isinstance(K, int) or isinstance(K, bool) or K < 1 or K > graph.num_nodes:
        raise ValueError("K must be an integer between 1 and graph.num_nodes")
    if K == 1:
        return torch.zeros(graph.num_nodes, dtype = torch.long, device = device)

    coordinates = spectral_embedding(
        graph,
        K = K,
        device = device,
        normalized = True,
    )
    coordinates = torch.nan_to_num(coordinates)

    kmeans = KMeans(
        n_clusters = K,
        init_method = "k-means++",
        num_init = min(8, graph.num_nodes - 1),
        verbose = False,
    )
    result = kmeans(coordinates.unsqueeze(0))

    return result.labels.squeeze(0).to(device = device, dtype = torch.long)

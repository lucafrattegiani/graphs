#Torch data and computations
import torch
from torch_geometric.data import Data

#Networkx graph analysis
import networkx as nx
from torch_geometric.utils import to_networkx

#Utilities
from ..utils.measures import gini_index

def betweenness_centrality(graph: Data, normalized: bool = True, device: torch.device | str = "cpu", centralization: str = "none") -> torch.Tensor:
    """
    Computes betweenness centrality for nodes in the graph. Defined as:

    G_i = sum_{s ≠ i ≠ t} sigma_{st}(i) / sigma_{st} -> Non normalized
    G_i = 2 / ((n-1)(n-2)) * G_i -> Normalized 

    sigma_{st} = Number of shortest paths between nodes s and t
    sigma_{st}(i) = Number of shortest paths between nodes s and t that pass through node i
    n = Number of nodes in the graph

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    normalized : bool
        Whether to normalize the betweenness centrality.
    device : torch.device | str
        Device to perform computations on.
    centralization : str
        Type of centralization to compute {"none", "freeman", "gini"}.

    Returns:
    -------
    torch.Tensor
        Betweenness centrality of the graph.
    """
    if centralization not in ["none", "freeman", "gini"]:
        raise ValueError("Centralization must be one of {'none', 'freeman', 'gini'}")

    if graph.num_nodes < 3: #Nan values for graphs with less than 3 nodes
        if centralization == "freeman" or centralization == "gini":
            return torch.tensor(torch.nan, dtype = torch.float32, device = device)
        else:
            return torch.full((graph.num_nodes,), torch.nan, dtype = torch.float32, device = device)

    # Compute betweenness centrality for each node
    graph_nx = to_networkx(graph, node_attrs = None, edge_attrs = None, to_undirected = True, remove_self_loops = True) #Convert to networkx graph for shortest path computation
    beetweeness_nx = nx.betweenness_centrality(graph_nx, weight = None, normalized = normalized) #Compute shortest paths through Bradnes
    betweeness = torch.tensor([beetweeness_nx[j] for j in range(graph.num_nodes)], dtype = torch.float32, device = device)

    if normalized: #Normalized betweenness centrality
        if centralization == "freeman": #Compute graph-level Freeman centralization
            max_betweeness = betweeness.max()
            betweeness = (max_betweeness - betweeness).sum() / (graph.num_nodes - 1)
        elif centralization == "gini": #Compute graph-level Gini centralization
            betweeness = gini_index(betweeness)
    else:
        if centralization == "freeman": #Compute graph-level Freeman centralization
            max_betweeness = betweeness.max()
            betweeness = 2 * (max_betweeness - betweeness).sum() / ((graph.num_nodes - 1)**2 * (graph.num_nodes - 2))

    return betweeness

def harmonic_centrality(graph: Data, device: torch.device | str = "cpu", normalized: bool = True, centralization: str = "none") -> torch.Tensor:
    """
    Computes the harmonic centrality of the graph. Defined as:

    H_i = sum_{j ≠ i} 1 / d(i, j) -> Non normalized
    H_i = 1 / (n-1) * H_i -> Normalized

    d(i, j) = Shortest path distance between nodes i and j
    n = Number of nodes in the graph

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    normalized : bool
        Whether to normalize the harmonic centrality.
    centralization : str
        Type of centralization to compute {"none", "freeman", "gini"}.

    Returns:
    -------
    torch.Tensor
        Harmonic centrality of the graph.
    """
    if centralization not in ["none", "freeman", "gini"]:
        raise ValueError("Centralization must be one of {'none', 'freeman', 'gini'}")

    if graph.num_nodes < 3: #Nan values for graphs with less than 3 nodes
        if centralization == "freeman":
            return torch.tensor(torch.nan, dtype = torch.float32, device = device)
        elif centralization == "gini":
            return torch.tensor(torch.nan, dtype = torch.float32, device = device)
        else:
            return torch.full((graph.num_nodes,), torch.nan, dtype = torch.float32, device = device)

    # Compute harmonic centrality for each node
    graph_nx = to_networkx(graph, node_attrs = None, edge_attrs = None, to_undirected = True, remove_self_loops = True) #Convert to networkx graph for shortest path computation
    harmonic_nx = nx.harmonic_centrality(graph_nx) #Compute shortest paths through Bradnes
    harmonic_centrality = torch.tensor([harmonic_nx[j] for j in range(graph.num_nodes)], dtype = torch.float32, device = device)

    if normalized: #Normalize harmonic centrality
        harmonic_centrality = harmonic_centrality / (graph.num_nodes - 1)
        if centralization == "freeman": #Compute graph-level Freeman centralization
            max_harmonic = harmonic_centrality.max()
            harmonic_centrality = 2 * (max_harmonic - harmonic_centrality).sum() / (graph.num_nodes - 2)
        elif centralization == "gini": #Compute graph-level Gini centralization
            harmonic_centrality = gini_index(harmonic_centrality)
    else:
        if centralization == "freeman": #Compute graph-level Freeman centralization
            max_harmonic = harmonic_centrality.max()
            harmonic_centrality = 2 * (max_harmonic - harmonic_centrality).sum() / ((graph.num_nodes - 1) * (graph.num_nodes - 2))
        elif centralization == "gini": #Compute graph-level Gini centralization
            harmonic_centrality = gini_index(harmonic_centrality)

    return harmonic_centrality

def pagerank_centrality(graph: Data, device: torch.device | str = "cpu", alpha: float = 0.85, directed: bool = False, centralization: str = "none") -> torch.Tensor:
    """
    Computes the PageRank centrality of the graph. Defined as:

    PR(i) = (1 - alpha) / n + alpha * sum_{j -> i} PR(j) / d(j)

    alpha = Damping factor
    n = Number of nodes in the graph
    d(j) = Out-degree of node j

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    alpha : float
        Damping factor.
    directed : bool
        Whether to treat the graph as directed or undirected.
    centralization : str
        Type of centralization to compute. Must be one of {'none', 'entropy', 'gini'}.

    Returns:
    -------
    torch.Tensor
        PageRank centrality of the graph.
    """
    if centralization not in ["none", "entropy", "gini"]:
        raise ValueError("Centralization must be one of {'none', 'entropy', 'gini'}")

    if graph.num_nodes < 2:
        return torch.full((graph.num_nodes,), torch.nan, dtype = torch.float32, device = device)

    # Compute PageRank centrality for each node
    graph_nx = to_networkx(graph, node_attrs = None, edge_attrs = None, to_undirected = not directed, remove_self_loops = True) #Convert to networkx graph for shortest path computation
    pagerank_nx = nx.pagerank(graph_nx, alpha = alpha) #Compute shortest paths through Bradnes
    pagerank_centrality = torch.tensor([pagerank_nx[j] for j in range(graph.num_nodes)], dtype = torch.float32, device = device)

    if centralization == "entropy":
        # Compute graph-level entropy centralization
        pagerank_centrality = -torch.sum(pagerank_centrality * torch.log2(pagerank_centrality + 1e-8))
        pagerank_centrality = 1 - pagerank_centrality / torch.log2(torch.tensor(graph.num_nodes, dtype = torch.float32, device = device)) #Normalize in [0, 1]
    elif centralization == "gini":
        # Compute graph-level Gini centralization
        pagerank_centrality = gini_index(pagerank_centrality)

    return pagerank_centrality

def centrality(graph: Data, method: str, device: torch.device | str = "cpu",
               centralization: str = "none", **kwargs) -> torch.Tensor:
    """
    Computes a centrality measure using the requested method.

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    method : str
        Centrality method to use: {"betwenness", "harmonic", "pagerank"}.
    device : torch.device | str
        Device to perform computations on.
    centralization : str
        Type of graph-level centralization. The accepted values depend on the
        selected method.
    **kwargs
        Additional arguments accepted by the selected centrality function.
        Use ``normalized`` for betweenness or harmonic centrality, and
        ``alpha`` or ``directed`` for PageRank.

    Returns:
    -------
    torch.Tensor
        The node-level centrality values or the requested graph-level
        centralization.

    Examples
    --------
    ``centrality(graph, "betwenness", normalized = False)``

    ``centrality(graph, "pagerank", alpha = 0.9, directed = True)``
    """
    centrality_functions = {
        "betwenness": betweenness_centrality,
        "harmonic": harmonic_centrality,
        "pagerank": pagerank_centrality,
    }
    if method not in centrality_functions:
        raise ValueError(
            "method must be one of {'betwenness', 'harmonic', 'pagerank'}"
        )

    return centrality_functions[method](
        graph,
        device = device,
        centralization = centralization,
        **kwargs,
    )

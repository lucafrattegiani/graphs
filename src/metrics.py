#Torch data and computations
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

#Networkx graph analysis
import networkx as nx
from torch_geometric.utils import to_networkx

#--------------------------------------------------
#DENSITY METRICS
#--------------------------------------------------

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
    if graph.num_nodes < 2:
        raise ValueError("Can't compute density of a 1 node graph")
    density = graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1))
    return density

#--------------------------------------------------
#CLUSTERING METRICS
#--------------------------------------------------

def local_clustering_coeff(graph: Data, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Computes the local clustering coefficient for each node in the graph. Defined as:

    C_i = 2T_i / (k_i(k_i - 1))

    T_i = Number of triangles that include node i
    k_i = Degree of node i

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.

    Returns:
    -------
    torch.Tensor
        Local clustering coefficient for each node.
    """
    if graph.num_nodes < 2:
        raise ValueError("Can't compute local clustering coefficient of a 1 node graph")
    
    # Adjacency matrix
    adj_matrix = torch.zeros((graph.num_nodes, graph.num_nodes), device = device)
    adj_matrix[graph.edge_index[0], graph.edge_index[1]] = 1

    # Compute local clustering coefficient
    triangles = torch.matmul(torch.matmul(adj_matrix, adj_matrix), adj_matrix).diagonal() # Compute number of triangles for each node
    degree = adj_matrix.sum(dim = 1) # Degree of each node
    
    # Avoid division by zero
    degree[degree < 2] = 2
    
    local_clustering = triangles / (degree * (degree - 1)) #Coefficient
    
    return local_clustering

def global_clustering_coeff(graph: Data, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Computes the global clustering coefficient of the graph. Defined as:

    C = T_{real} / T_{possible}

    T_{real} = Total number of triangles in the graph
    T_{possible} = Sum of the number of possibly existing triangles for each node i having degree k_i, which is k_i(k_i - 1) / 2

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    """
    if graph.num_nodes < 3:
        raise ValueError("Can't compute global clustering coefficient of a graph with less than 3 nodes")

    # Adjacency matrix
    adj_matrix = torch.zeros((graph.num_nodes, graph.num_nodes), device = device)
    adj_matrix[graph.edge_index[0], graph.edge_index[1]] = 1

    # Compute global clustering coefficient
    real_triangles = torch.matmul(torch.matmul(adj_matrix, adj_matrix), adj_matrix).diagonal().sum() # Total number of triangles
    degree = adj_matrix.sum(dim = 1) # Degree of each node
    possible_triangles = (degree * (degree - 1)).sum() # Total number of possible triangles per node

    # Avoid division by zero
    if possible_triangles == 0:
        return torch.tensor(0.0, device=device)

    global_clustering = real_triangles / possible_triangles

    return global_clustering

#--------------------------------------------------
#CENTRALITY METRICS
#--------------------------------------------------

def freeman_centrality(graph: Data, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Computes the Freeman centrality of the graph. Defined as:

    C_F = sum_{i=1}^{n} (C_{max} - C_i) / (n-1)(n-2)

    C_{max} = Maximum degree centrality in the graph
    C_i = Degree centrality of node i
    n = Number of nodes in the graph

    So it is measures a ratio in [0, 1] computing differences between the maximum degree centrality and the degree centrality of each node, 
    normalized by the number of nodes in the graph.

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.

    Returns:
    -------
    torch.Tensor
        Freeman centrality of the graph.
    """
    if graph.num_nodes < 3:
        raise ValueError("Can't compute Freeman centrality of a graph with less than 3 nodes")

    # Compute degree centrality for each node
    degree_centrality = degree(
        graph.edge_index[0],
        num_nodes = graph.num_nodes,
        dtype = torch.float32,
    ).to(device)

    # Compute maximum degree centrality
    max_degree_centrality = degree_centrality.max()

    # Compute Freeman centrality
    freeman_centrality_value = (max_degree_centrality - degree_centrality).sum() / ((graph.num_nodes - 1) * (graph.num_nodes - 2))

    return freeman_centrality_value

def gini_betweenness(graph: Data, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Computes the Gini coefficient of the betweenness centrality of the graph. Defined as:

    G = sum_{i=1}^{n} sum_{j=1}^{n} |C_i - C_j| / (2n * sum_{i=1}^{n} C_i)

    C_i = Betweenness centrality of node i
    n = Number of nodes in the graph

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.

    Returns:
    -------
    torch.Tensor
        Gini coefficient of the betweenness centrality of the graph.
    """
    if graph.num_nodes < 2:
        raise ValueError("Can't compute Gini coefficient of a graph with less than 2 nodes")

    # Compute betweenness centrality for each node
    graph_nx = to_networkx(graph, node_attrs = None, edge_attrs = None, to_undirected = True) #Convert to networkx graph for shortest path computation
    beetweeness_nx = nx.betweenness_centrality(graph_nx, weight = None, normalized = False) #Compute shortest paths through Bradnes
    betweeness = torch.tensor([beetweeness_nx[j] for j in range(graph.num_nodes)], dtype = torch.float32, device = device)

    # Compute Gini coefficient
    ordered = torch.sort(betweeness).values
    total = ordered.sum()
    if total > 0:  
        n = len(ordered)
        cumulative = torch.cumsum(ordered / total, dim = 0)
        gini_coefficient = (n + 1 - 2 * (cumulative[:-1]).sum()) / n
    else:
        gini_coefficient = torch.tensor(0.0, device = device)

    return gini_coefficient

def gini_harmonic(graph: Data, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Computes the Gini coefficient of the harmonic centrality of the graph. Defined as:

    G = sum_{i=1}^{n} sum_{j=1}^{n} |C_i - C_j| / (2n * sum_{i=1}^{n} C_i)

    C_i = Harmonic centrality of node i
    n = Number of nodes in the graph

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.

    Returns:
    -------
    torch.Tensor
        Gini coefficient of the harmonic centrality of the graph.
    """
    if graph.num_nodes < 2:
        raise ValueError("Can't compute Gini coefficient of a graph with less than 2 nodes")

    # Compute harmonic centrality for each node
    graph_nx = to_networkx(graph, node_attrs = None, edge_attrs = None, to_undirected = True) #Convert to networkx graph for shortest path computation
    harmonic_nx = nx.harmonic_centrality(graph_nx) #Compute shortest paths through Bradnes
    harmonic_centrality = torch.tensor([harmonic_nx[j] for j in range(graph.num_nodes)], dtype = torch.float32, device = device)

    # Compute Gini coefficient
    ordered = torch.sort(harmonic_centrality).values
    total = ordered.sum()
    if total > 0:
        n = len(ordered)
        cumulative = torch.cumsum(ordered / total, dim = 0)
        gini_coefficient = (n + 1 - 2 * (cumulative[:-1]).sum()) / n
    else:
        gini_coefficient = torch.tensor(0.0, device = device)

    return gini_coefficient

#--------------------------------------------------
#HOMOPHILY METRICS
#--------------------------------------------------

def jsd_homophily(graph: Data, device: torch.device | str = "cpu", eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the edge-wise Jensen-Shannon divergence homophily of the graph. Defined as:

    JSD(P || Q) = 1/2 * (KL(P || M) + KL(Q || M))

    P = Distribution of the source node attributes
    Q = Distribution of the target node attributes
    M = 1/2 * (P + Q)

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    eps : float
        Small value to avoid division by zero.

    Returns:
    -------
    torch.Tensor
        Mean Jensen-Shannon divergence over the graph edges.
    """
    if not hasattr(graph, "x") or graph.x is None:
        raise ValueError("Graph must have node attributes to compute homophily")
    if graph.edge_index.numel() == 0:
        return torch.tensor(0.0, device = device)

    attributes = graph.x.to(device = device, dtype = torch.float32)
    if attributes.dim() != 2:
        raise ValueError("Node attributes must be a 2D tensor")
    if (attributes < 0).any():
        raise ValueError("JSD requires non-negative node attribute distributions")

    #Normalize node attributes as probability distributions
    attributes = attributes.clamp(min = eps)
    attributes = attributes / attributes.sum(dim = 1, keepdim = True).clamp(min = eps)

    #Edges sources/ends
    source, target = graph.edge_index.to(device)
    start = attributes[source]
    end = attributes[target]
    mixture = 0.5 * (start + end)

    #JSD divergence
    jsd = 0.5 * (start * torch.log2(start / mixture)).sum(dim = 1) #Edge-wise source term
    jsd = jsd + 0.5 * (end * torch.log2(end / mixture)).sum(dim = 1) #Edge-wise target term

    return jsd.mean()

def adjusted_homophily(graph: Data, device: torch.device | str = "cpu", eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the adjusted homophily of an undirected and unweighted graph. Defined as:

    H_adj = (H_observed - H_expected) / (1 - H_expected)

    H_observed = Fraction of edges connecting nodes with the same discrete attribute
    H_expected = Expected homophily from the edge-endpoint attribute distribution

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data. It must have one discrete node attribute per node in graph.x.
    device : torch.device | str
        Device to perform computations on.
    eps : float
        Small value to avoid division by zero.

    Returns:
    -------
    torch.Tensor
        Adjusted homophily of the graph.
    """
    if not hasattr(graph, "x") or graph.x is None:
        raise ValueError("Graph must have node attributes to compute adjusted homophily")
    if graph.edge_index.numel() == 0:
        return torch.tensor(0.0, device = device)

    attributes = graph.x.to(device)
    if attributes.dim() == 2 and attributes.size(1) == 1:
        attributes = attributes.squeeze(1)
    elif attributes.dim() != 1:
        raise ValueError("Adjusted homophily requires one discrete node attribute per node")
    if attributes.numel() != graph.num_nodes:
        raise ValueError("Number of node attributes must match graph.num_nodes")

    source, target = graph.edge_index.to(device)
    no_self_loops = source != target
    source = source[no_self_loops]
    target = target[no_self_loops]
    if source.numel() == 0:
        return torch.tensor(0.0, device = device)

    source_attributes = attributes[source]
    target_attributes = attributes[target]

    observed_homophily = (source_attributes == target_attributes).float().mean()

    _, endpoint_classes = torch.unique(
        torch.cat([source_attributes, target_attributes]),
        sorted = True,
        return_inverse = True,
    )
    class_counts = torch.bincount(endpoint_classes).to(dtype = torch.float32, device = device)
    class_probabilities = class_counts / class_counts.sum().clamp(min = eps)
    expected_homophily = (class_probabilities ** 2).sum()

    denominator = 1.0 - expected_homophily
    if denominator.abs() < eps:
        return torch.tensor(0.0, device = device)

    return (observed_homophily - expected_homophily) / denominator

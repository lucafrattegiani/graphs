#Torch data and computations
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

#Networkx graph analysis
import networkx as nx
from torch_geometric.utils import to_networkx, to_dense_adj

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
    if graph.num_nodes < 2: #Nan values for 1 node graphs
        return torch.full((graph.num_nodes,), torch.nan, dtype = torch.float32, device = device)
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
    if graph.num_nodes < 2: #Nan values for 1 node graphs
        return torch.full((graph.num_nodes,), torch.nan, dtype = torch.float32, device = device)
    
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

    C = T_{closed} / T_{total}

    T_{closed} = Total number of closed triplets in the graph
    T_{total} = Total number of triplets in the graph

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    """
    if graph.num_nodes < 3: #Nan values for graphs with less than 3 nodes
        return torch.tensor(torch.nan, dtype = torch.float32, device = device)

    # Adjacency matrix
    adj_matrix = torch.zeros((graph.num_nodes, graph.num_nodes), device = device)
    adj_matrix[graph.edge_index[0], graph.edge_index[1]] = 1

    # Compute global clustering coefficient
    closed_triplets = torch.matmul(torch.matmul(adj_matrix, adj_matrix), adj_matrix).diagonal().sum() # Total number of closed triplets
    degree = adj_matrix.sum() # Sum of degrees (self 2-edges paths)
    norm = torch.matmul(adj_matrix, adj_matrix).sum() # Total number of 2-edges paths
    total_triplets =  (norm - degree)# Total number of possible triplets

    # Avoid division by zero
    if total_triplets == 0:
        return torch.tensor(0.0, device=device)

    global_clustering = closed_triplets / total_triplets

    return global_clustering

#--------------------------------------------------
#CENTRALITY METRICS
#--------------------------------------------------

def gini_index(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gini index for a vector of positive measuraments.

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor.

    Returns:
    -------
    torch.Tensor
        Gini index of the input vector.
    """
    n = x.numel() #Vector length
    if n == 0: #Nan values for empty tensors
        return torch.tensor(torch.nan, device=x.device)
    
    # Sort the measurements in ascending order
    sorted_x = torch.sort(x)[0]
    
    # Compute the Gini index
    cdf = torch.cumsum(sorted_x, dim = 0) / sorted_x.sum()
    gini_index = 1 - (1/n) * (cdf.sum() + cdf[:-1].sum())
    
    return gini_index

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

#--------------------------------------------------
#HOMOPHILY AND INFORMATIVITY METRICS
#--------------------------------------------------

def jsd_informativity(graph: Data, device: torch.device | str = "cpu", aggregate: bool = True) -> torch.Tensor:
    """
    Computes the Jensen-Shannon informativity coefficients of the graph for discrete node labels in {1, ..., K}. Defined as:

    rho(k) = JSD(P(k) || Q) -> Non aggregated, for any class k
    rho = sum_{k} (n_k / n) * rho(k) -> Aggregated over all classes

    P(k) = Label's distribution among neighborhoods of nodes belonging to class k
    Q = Label's distribution among neighborhoods of a generic node without any class conditioning
    n_k = Number of nodes belonging to class k
    n = Total number of nodes in the graph

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    aggregate : bool
        Whether to aggregate the Jensen-Shannon divergence over multiple classes.

    Returns:
    -------
    torch.Tensor
        Jensen-Shannon coefficients for neighborhood informativeness.
    """
    # Check correctness of node attributes
    if not hasattr(graph, "x") or graph.x is None:
        raise ValueError("Graph must have node attributes to compute JSD informativeness")
    if graph.x.dim() > 1 and graph.x.shape[1] != 1 and graph.x.numel() != graph.num_nodes:
        raise ValueError("Node attributes must be a single discrete label per node")
    
    # Check if graph has nodes
    if graph.num_nodes < 2:
        return torch.tensor(torch.nan, device = device)
    
    #Compute the marginal distribution
    labels = graph.x.view(-1).long().to(device) #Extract labels as 1D vector and send to device
    edges = graph.edge_index.to(device) #Extract edge set and send to device
    n = graph.num_nodes #Number of nodes
    K = labels.max().item() + 1 #Number of classes

    degrees = torch.bincount(edges[0], minlength = n).to(dtype = torch.float32, device = device) #Compute node degrees
    q = torch.zeros(K, dtype = torch.float32, device = device) #Initialize marginal distribution

    #Compute the marginal distribution q
    q.index_add_(0, labels, degrees)
    q = q / degrees.sum() # CORREZIONE 1: Divisione per la somma reale dei gradi e assegnazione esplicita

    #Compute one hot encoded vectors and sum neighboring labels for each node
    one_hot = torch.nn.functional.one_hot(labels, num_classes = K).float() #One-hot enoding
    neighborhoods = torch.zeros((n, K), dtype = torch.float32, device = device) #Initialize neighborhood counts for any node
    neighborhoods.index_add_(0, edges[0], one_hot[edges[1]]) #Count neighbors one hot encoded labels for each node

    #Avoid division by 0
    degrees_safe = degrees.clamp(min = 1.0)
    normalization = degrees_safe**(-1)
    normalization[degrees == 0] = 0.0 #Set normalization to 0 for nodes with no neighbors

    #Normalize neighborhood counts
    neighborhoods = neighborhoods * normalization.unsqueeze(1) #Normalize neighborhood counts to get neighborhood distributions
    p_k = torch.zeros((K, K), dtype = torch.float32, device = device) #Initialize class conditioned neighborhood distributions
    p_k.index_add_(0, labels, neighborhoods) #Count neighborhood distributions for each class
    n_k = torch.bincount(labels, minlength = K).to(dtype = torch.float32, device = device) #Count number of nodes for each class
    n_k_safe = n_k.clamp(min = 1.0) #Avoid division by 0
    normalization = n_k_safe**(-1)
    normalization[n_k == 0] = 0.0 #Set normalization to 0
    p_k = p_k * normalization.unsqueeze(1) #Normalize class conditioned neighborhood distributions

    #Compute jsd divergences
    q = q.unsqueeze(0) #Reshape q to (1, K)
    M = 0.5 * (p_k + q) #Mixture distribution

    #Avoid zero-terms
    q_safe = q.clamp(min = 1e-8)
    p_k_safe = p_k.clamp(min = 1e-8)
    M_safe = M.clamp(min = 1e-8)

    #KL divergences
    kl_p_m = (p_k * (torch.log2(p_k_safe) - torch.log2(M_safe))).sum(dim = 1) #KL(P || M)
    
    # CORREZIONE 2: Uso della funzione di broadcasting .sum() in modo coerente
    kl_q_m = (q * (torch.log2(q_safe) - torch.log2(M_safe))).sum(dim = 1) #KL(Q || M)

    #JS divergence per class
    jsd_k = 0.5 * (kl_p_m + kl_q_m) #JSD(P || Q)

    # CORREZIONE 3: Inversione dei calcoli logaritmici se la classe è vuota (evita NaN in output)
    jsd_k[n_k == 0] = 0.0

    if aggregate:
        #Aggregate JSD over classes
        jsd = (n_k * jsd_k).sum() / n #Weighted average of JSD over classes
        return jsd
    else:
        return jsd_k 

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

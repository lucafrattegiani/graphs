#Torch data and computations
import torch
from torch_geometric.data import Data
from ..validity import _graph_device, _floating_dtype

#--------------------------------------------------
#EDGE HOMOPHILY
#--------------------------------------------------

def hard_edge_homophily(graph: Data, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Computes the edge-wise homophily of an undirected graph with hard node attributes. Defined as:

    H = 1 / |E| * sum_{(i, j) in E} 1_{x_i == x_j}

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Computation and output device. If None, edge_index and x must
        share a device. An explicit device permits local transfers; the
        graph is not modified. Floating feature precision is preserved.

    Returns:
    -------
    torch.Tensor
        Mean homophily over the graph edges.
    """
    device = _graph_device(graph, device, attributes = ("edge_index", "x"))
    labels = graph.x.long().to(device = device)
    source, target = graph.edge_index.to(device)
    
    # Compute edge-wise homophily
    edge_homophily = (labels[source] * labels[target]).to(_floating_dtype(graph.x)).sum(dim = 1).mean()
    
    return edge_homophily

def soft_edge_homophily(graph: Data, device: torch.device | str | None = None, eps: float = 1e-8, method: str = "jsd") -> torch.Tensor:
    """
    Computes the edge-wise Jensen-Shannon divergence homophily of an undirected graph with soft node attributes. Defined as:

    h = 1 / |E| * sum_{(i, j) in E} rho(x_i, x_j)

    where:

    rho(x_i, x_j) = JSD(x_i || x_j) = 1/2 * (KL(x_i || M) + KL(x_j || M)) -> If method == "jsd"
    rho(x_i, x_j) = <x_i, x_j> -> If method == "euclidean"

    x_i = Distribution of the source node attributes
    x_j = Distribution of the target node attributes
    M = 1/2 * (x_i + x_j)

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Computation and output device. If None, edge_index and x must
        share a device. An explicit device permits local transfers; the
        graph is not modified. Floating feature precision is preserved.
    eps : float
        Small value to avoid division by zero.
    method : str
        Method to compute the edge-wise homophily {"jsd", "euclidean"}.

    Returns:
    -------
    torch.Tensor
        Mean Jensen-Shannon divergence over the graph edges.
    """
    device = _graph_device(graph, device, attributes = ("edge_index", "x"))
    attributes = graph.x.to(device = device, dtype = _floating_dtype(graph.x))
    if (attributes < 0).any():
        raise ValueError("JSD requires non-negative node attribute distributions")

    if method not in ["jsd", "euclidean"]:
        raise ValueError("Method must be one of {'jsd', 'euclidean'}")

    if method == "euclidean":
        source, target = graph.edge_index.to(device)
        start = attributes[source]
        end = attributes[target]
        edge_homophily = (start * end).sum(dim = 1)
        return edge_homophily.mean()
    elif method == "jsd":
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

def continuous_edge_homophily(graph: Data, device: torch.device | str | None = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the edge-wise homophily of an undirected graph with continuous node attributes. Defined as:

    \tilde{x}_i = (x_i - \bar{x}) / sigma -> Standardized node attributes
    H = 1 / |E| * sum_{(i, j) in E} < \tilde{x}_i - \tilde{x}_j > -> Continuous edge homophily

    where:
    \bar{x} = Empirical mean of node attributes
    sigma = Empirical standard deviation of node attributes

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Computation and output device. If None, edge_index and x must
        share a device. An explicit device permits local transfers; the
        graph is not modified. Floating feature precision is preserved.
    eps : float
        Small value to avoid division by zero.

    Returns:
    -------
    torch.Tensor
        Mean homophily over the graph edges.
    """
    device = _graph_device(graph, device, attributes = ("edge_index", "x"))
    attributes = graph.x.to(device = device, dtype = _floating_dtype(graph.x))
    standardized = (attributes - attributes.mean(dim = 0, keepdim = True)) / (attributes.std(dim = 0, keepdim = True) + eps) #Standardize features
    
    #Edges sources/ends
    source, target = graph.edge_index.to(device)
    start = standardized[source]
    end = standardized[target]

    edge_homophily = (start * end).mean()

    return edge_homophily

def edge_homophily(graph: Data, device: torch.device | str | None = None,
                   attributes_type: str = "hard", **kwargs) -> torch.Tensor:
    """
    Computes the edge-wise homophily of an undirected graph depending on node attributes.

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Computation and output device. If None, edge_index and x must
        share a device. An explicit device permits local transfers; the
        graph is not modified. Floating feature precision is preserved.
    attributes_type : str
        Type of node attributes {"hard", "soft", "continuous"}.
    **kwargs
        Additional arguments accepted by the selected homophily function.
        Use ``eps`` for soft or continuous attributes and ``method`` for soft
        attributes.

    Returns:
    -------
    torch.Tensor
        Edge-wise homophily of the graph.
    """
    device = _graph_device(graph, device, attributes = ("edge_index", "x"))
    # Check if graph is valid
    if graph.num_nodes < 2 or graph.edge_index is None or graph.edge_index.size(1) == 0:
        return torch.tensor(torch.nan, dtype = _floating_dtype(graph.x), device = device)
    if not hasattr(graph, "x") or graph.x is None:
        raise ValueError("Graph must have node attributes to compute edge homophily")
    if graph.x.dim() != 2:
        raise ValueError((f"Errore shape: expected 2D [n, K], received {graph.x.dim()}D with shape {graph.x.shape}"))
    homophily_functions = {
        "hard": hard_edge_homophily,
        "soft": soft_edge_homophily,
        "continuous": continuous_edge_homophily,
    }
    if attributes_type not in homophily_functions:
        raise ValueError("attributes_type must be one of {'hard', 'soft', 'continuous'}")

    return homophily_functions[attributes_type](
        graph,
        device = device,
        **kwargs,
    )

#--------------------------------------------------
#ADJUSTED HOMOPHILY
#--------------------------------------------------

def adjusted_homophily_hard(graph: Data, device: torch.device | str | None = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the adjusted homophily of an undirected graph with discrete node attributes. 
    Defined as:

    H_adj = (H_observed - H_expected) / (1 - H_expected)

    H_observed = Fraction of edges connecting nodes with same attribute
    H_expected = Expected homophily given the node attribute distribution and degrees

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data. graph.x must contain one-hot class labels, one row per node.
    device : torch.device | str
        Computation and output device. If None, edge_index and x must
        share a device. An explicit device permits local transfers; the
        graph is not modified. Floating feature precision is preserved.
    eps : float
        Small value to avoid division by zero when H_expected == 1.

    Returns:
    -------
    torch.Tensor
        Adjusted homophily of the graph.
    """        
    device = _graph_device(graph, device, attributes = ("edge_index", "x"))
    # Extract data and send to device
    attributes = graph.x.to(device = device, dtype = torch.long)
    # One-hot rows identify exactly one class; keep class extraction on device.
    labels = attributes.argmax(dim = 1)
    edges = graph.edge_index.to(device)
    n = graph.num_nodes
    num_edges = edges.size(1)
    K = attributes.shape[1] # Includes unused classes without reading a GPU scalar

    # Compute node degrees 
    dtype = _floating_dtype(graph.x)
    degrees = torch.bincount(edges[0], minlength=n).to(dtype=dtype)
    
    # Compute marginal degree-weighted label distribution Q(k)
    q = torch.zeros(K, dtype=dtype, device=device)
    q.index_add_(0, labels, degrees)
    q = q / num_edges

    # Expected homophily (Chung-Lu null model)
    H_e = (q ** 2).sum()

    # Observed homophily (Mean of intra-class edge indicators)
    # Reuse class indices and edges instead of preparing the graph a second time.
    source, target = edges
    H_o = (labels[source] == labels[target]).to(dtype).mean()

    # Adjusted homophily
    H_adj = (H_o - H_e) / (1.0 - H_e + eps)

    return H_adj

def adjusted_homophily_soft(graph: Data, device: torch.device | str | None = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the adjusted homophily of a directed/undirected graph with soft node attributes. 
    Defined as:

    H_adj = (H_observed - H_expected) / (1 - H_expected)

    H_observed = Fraction of edges connecting nodes with same attribute
    H_expected = Expected homophily given the node attribute distribution and degrees

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data. It must have one soft node attribute per node in graph.x.
    device : torch.device | str
        Computation and output device. If None, edge_index and x must
        share a device. An explicit device permits local transfers; the
        graph is not modified. Floating feature precision is preserved.
    eps : float
        Small value to avoid division by zero when H_expected == 1.

    Returns:
    -------
    torch.Tensor
        Adjusted homophily of the graph.
    """
    device = _graph_device(graph, device, attributes = ("edge_index", "x"))
    if not hasattr(graph, "x") or graph.x is None:
        raise ValueError("Graph must have node attributes to compute Adjusted Homophily")
    
    if graph.num_nodes < 2 or graph.edge_index is None or graph.edge_index.size(1) == 0:
        return torch.tensor(torch.nan, dtype = _floating_dtype(graph.x), device = device)
    
    labels = graph.x.to(device = device, dtype = _floating_dtype(graph.x))
    # Preserve the non-negativity check previously performed by the soft helper.
    if (labels < 0).any():
        raise ValueError("JSD requires non-negative node attribute distributions")
    edges = graph.edge_index.to(device)
    n = graph.num_nodes
    num_edges = edges.size(1)

    # Node degrees
    degrees = torch.bincount(edges[0], minlength=n).to(labels.dtype)

    # Null disttribution 
    q = torch.mv(labels.t(), degrees) / num_edges

    # Expected homophily
    H_e = torch.dot(q, q)

    # Reuse prepared tensors: each edge contributes the dot product of its labels.
    source, target = edges
    H_o = (labels[source] * labels[target]).sum(dim = 1).mean()

    H_adj = (H_o - H_e) / (1.0 - H_e + eps)

    return H_adj

def adjusted_homophily(graph: Data, device: torch.device | str | None = None,
                       eps: float = 1e-8, attributes_type: str = "hard") -> torch.Tensor:
    """
    Computes the adjusted homophily of an undirected graph with hard or soft node attributes.
    Defined as:

    H_adj = (H_observed - H_expected) / (1 - H_expected)

    H_observed = Fraction of edges connecting nodes with same attribute
    H_expected = Expected homophily given the node attribute distribution and degrees

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data. It must have one discrete or soft node attribute per node in graph.x.
    device : torch.device | str
        Computation and output device. If None, edge_index and x must
        share a device. An explicit device permits local transfers; the
        graph is not modified. Floating feature precision is preserved.
    eps : float
        Small value to avoid division by zero when H_expected == 1.
    attributes_type : str
        Type of node attributes {"hard", "soft"}.
    **kwargs
        Additional arguments accepted by the selected adjusted-homophily
        function.

    Returns:
    -------
    torch.Tensor
        Adjusted homophily of the graph.
    """
    device = _graph_device(graph, device, attributes = ("edge_index", "x"))
    # Check if graph is valid
    if graph.num_nodes < 2 or graph.edge_index is None or graph.edge_index.size(1) == 0:
        return torch.tensor(torch.nan, dtype = _floating_dtype(graph.x), device = device)
    if graph.x.dim() != 2:
        raise ValueError((f"Errore shape: expected 2D [n, K], received {graph.x.dim()}D with shape {graph.x.shape}"))
    adjusted_homophily_functions = {
        "hard": adjusted_homophily_hard,
        "soft": adjusted_homophily_soft,
    }
    if attributes_type not in adjusted_homophily_functions:
        raise ValueError("attributes_type must be one of {'hard', 'soft'}")

    return adjusted_homophily_functions[attributes_type](
        graph,
        device = device,
        eps = eps,
    )

#--------------------------------------------------
#LABEL SMOOTHNESS
#--------------------------------------------------

def dirichlet_energy(graph: Data, aggregate: bool = True,
                     device: torch.device | str | None = None) -> torch.Tensor:
    """
    Computes the Dirichlet energy of an undirected graph depending on node attributes. Defined as:

    D(e) = 0.5 ||x_i - x_j||^2 -> Edge-level
    D(G) = 1 / |E| sum_{(i, j) in E} D(e) -> Graph-level

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    aggregate : bool
        Whether to aggregate the Dirichlet energy to get a graph-level measure.

    Returns:
    -------
    torch.Tensor
        Dirichlet energy of the graph.
    """
    device = _graph_device(graph, device, attributes = ("edge_index", "x"))
    x = graph.x.to(device = device, dtype = _floating_dtype(graph.x))
    edge_index = graph.edge_index.to(device)
    src, dst = edge_index

    # Compute squared Euclidean distance for each edge
    sq_distances = torch.sum((x[src] - x[dst]) ** 2, dim=-1)

    # Standard Dirichlet energy includes a 0.5 factor
    edge_energies = 0.5 * sq_distances

    if aggregate:
        # Graph-level aggregation: mean over all directed edges
        return edge_energies.mean()
    else:
        # Edge-level energies
        return edge_energies

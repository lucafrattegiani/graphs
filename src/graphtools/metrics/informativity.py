#Torch data and computations
import torch
from torch_geometric.data import Data
from ..utils.measures import gaussian_hellinger_distance, gaussian_kl_divergence

def jsd_informativeness(graph: Data, device: torch.device | str = "cpu", aggregate: bool = True, attributes_type: str = "hard", level: str = "node") -> torch.Tensor:
    """
    Computes the Jensen-Shannon informativeness coefficients of the graph for discrete node labels in {1, ..., K} with either hard or soft assignments. Defined as:

    rho_i = JSD(P_i || Q) -> Node level
    rho_k = JSD(P^k || Q) -> Class level

    P_i = Label's distribution among neighborhoods of node i
    P^k = Label's distribution among neighborhoods of nodes belonging to class k
    Q = Label's distribution among neighborhoods of a generic node without any class/node conditioning

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    aggregate : bool
        Whether to aggregate the Jensen-Shannon to get a graph-level measure.
    attribute_type : str
        Type of node attributes {"hard", "soft"}.
    level : str
        Level at which to compute the informativity {"node", "class"}.

    Returns:
    -------
    torch.Tensor
        Jensen-Shannon coefficients for neighborhood informativeness.
    """
    if level == "class" and attributes_type == "soft":
        raise ValueError("Class-level informativity is only defined for hard node assignments")
    
    # Extract labels as K-dimensional vectors and send to device
    labels = graph.x.float().to(device) 
    edges = graph.edge_index.to(device) 
    n = graph.num_nodes 
    
    # Get the number of classes directly from the 2D tensor shape
    K = labels.size(1) 
    num_edges = edges.size(1)
    
    # Compute node degrees
    degrees = torch.bincount(edges[0], minlength=n).to(dtype=torch.float32, device=device) 

    # Compute the marginal distribution q
    q = (labels * degrees.view(-1, 1)).sum(dim=0)
    q = q / num_edges

    # Initialize neighborhood counts for any node
    p_i = torch.zeros((n, K), dtype=torch.float32, device=device) 
    # Count neighbors' one-hot encoded (or soft) labels for each node
    p_i.index_add_(0, edges[0], labels[edges[1]]) 

    # Avoid division by 0
    degrees_safe = degrees.clamp(min=1.0)
    normalization = degrees_safe**(-1)
    normalization[degrees == 0] = 0.0 # Set normalization to 0 for nodes with no neighbors

    # Normalize neighborhood counts to get neighborhood distributions
    p_i = p_i * normalization.unsqueeze(1) 
    
    if level == "node":
        # Compute JSD divergences for each node
        M = 0.5 * (p_i + q) # Mixture distribution

        # Avoid zero-terms
        q_safe = q.clamp(min=1e-8)
        p_i_safe = p_i.clamp(min=1e-8)
        M_safe = M.clamp(min=1e-8)

        # KL divergences
        kl_p_m = (p_i * (torch.log2(p_i_safe) - torch.log2(M_safe))).sum(dim=1) # KL(P || M)
        kl_q_m = (q * (torch.log2(q_safe) - torch.log2(M_safe))).sum(dim=1) # KL(Q || M)

        # JS divergence for each node
        jsd_i = 0.5 * (kl_p_m + kl_q_m) # JSD(P || Q)

        if aggregate:
            jsd = (degrees * jsd_i).sum() / num_edges # Degree-weighted average JSD over nodes
            return jsd
        else:
            return jsd_i
            
    elif level == "class":
        # Convert 2D labels to 1D scalar indices for class grouping
        labels_1d = labels.argmax(dim=1)
        
        # Initialize class conditioned neighborhood distributions
        p_k = torch.zeros((K, K), dtype=torch.float32, device=device) 
        # Count neighborhood distributions for each class
        p_k.index_add_(0, labels_1d, p_i) 
        
        # Count number of nodes for each class
        n_k = torch.bincount(labels_1d, minlength=K).to(dtype=torch.float32, device=device) 
        
        # Avoid division by 0
        n_k_safe = n_k.clamp(min=1.0) 
        normalization = n_k_safe**(-1)
        normalization[n_k == 0] = 0.0 # Set normalization to 0 for classes with no nodes
        
        # Normalize class conditioned neighborhood distributions
        p_k = p_k * normalization.unsqueeze(1) 

        # Compute JSD divergences for each class
        M = 0.5 * (p_k + q) # Mixture distribution

        # Avoid zero-terms
        q_safe = q.clamp(min=1e-8)
        p_k_safe = p_k.clamp(min=1e-8)
        M_safe = M.clamp(min=1e-8)

        # KL divergences
        kl_p_m = (p_k * (torch.log2(p_k_safe) - torch.log2(M_safe))).sum(dim=1) # KL(P || M)
        kl_q_m = (q * (torch.log2(q_safe) - torch.log2(M_safe))).sum(dim=1) # KL(Q || M)

        # JS divergence for each class
        jsd_k = 0.5 * (kl_p_m + kl_q_m) # JSD(P || Q)

        if aggregate:
            # Weighted average of JSD over classes
            jsd = (n_k * jsd_k).sum() / n 
            return jsd
        else:
            return jsd_k

def non_parametric_neighbor_informativeness(graph: Data, device: torch.device | str = "cpu", aggregate: bool = True) -> torch.Tensor:
    """
    Computes the neighborhood informativeness of a graph with continuous node attributes, through non-parametric methods.

    rho_i = Wasserstein-1(P_i, Q) -> Node level

    P_i = Empirical distribution of the continuous node attributes among the neighbors of node i
    Q = Empirical distribution of the continuous node attributes among all nodes in the graph

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    aggregate : bool
        Whether to aggregate the neighborhood informativeness to get a graph-level measure.

    Returns:
    -------
    torch.Tensor
        Non-parametric neighborhood informativeness of the graph.
    """
    raise NotImplementedError("Non-parametric neighborhood informativeness is not yet implemented. Please use the parametric method instead.")

def parametric_neighbor_informativeness(graph: Data, device: torch.device | str = "cpu", aggregate: bool = True, eps: float = 1e-8, diagonal: bool = False,
                                        distance: str = "KL") -> torch.Tensor:
    """
    Computes the neighborhood informativeness of a graph with continuous node attributes, through parametric methods.

    rho_i = D(P_i, Q) -> Node level

    P_i = Gaussian distribution fitted to the continuous node attributes among the neighbors of node i
    Q = Gaussian distribution fitted to the continuous node attributes among all nodes in the graph

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    aggregate : bool
        Whether to aggregate the neighborhood informativeness to get a graph-level measure.
    eps : float
        Small value to avoid division by zero.
    diagonal : bool
        Whether to assume diagonal covariance matrices for the Gaussian distributions.
    distance : str
        Distance between distributions {"KL", "hellinger"}.

    Returns:
    -------
    torch.Tensor
        Parametric neighborhood informativeness of the graph.
    """
    attributes = graph.x.to(device = device)
    source, target = graph.edge_index.to(device)
    degrees = torch.bincount(source, minlength = graph.num_nodes)

    # Compute degree-weighted global mean
    degree_weights = degrees.to(attributes.dtype)
    total_degree = degree_weights.sum()
    mu = (
        attributes * degree_weights.unsqueeze(1)
    ).sum(dim = 0, keepdim = True) / total_degree.clamp(min = 1)
    centered = attributes - mu
    weighted_centered = centered * degree_weights.unsqueeze(1)
    global_covariance_denominator = total_degree.clamp(min = 1)

    mu_i = torch.zeros((graph.num_nodes, graph.num_features), dtype = attributes.dtype, device = device)
    mu_i.index_add_(0, source, attributes[target])
    mu_i = mu_i / degrees.clamp(min = 1).to(attributes.dtype).unsqueeze(1)

    covariance_denominator = (degrees - 1).clamp(min = 1).to(attributes.dtype)
    neighbor_centered = attributes[target] - mu_i[source]

    # Select distribution distance function
    distance_key = distance.lower()
    if distance_key == "kl":
        distance_function = gaussian_kl_divergence
    elif distance_key == "hellinger":
        distance_function = gaussian_hellinger_distance
    else:
        raise ValueError("Distance must be one of {'KL', 'hellinger'}")

    # Diagonal covariance vs full covariance
    if diagonal:
        # Degree-weighted variance of each feature over all graph nodes
        variances = (
            (centered * weighted_centered).sum(dim = 0)
            / global_covariance_denominator
        ) + eps

        # Variance of each feature within every node neighborhood
        variances_i = torch.zeros(
            (graph.num_nodes, graph.num_features),
            dtype = attributes.dtype,
            device = device,
        )
        variances_i.index_add_(0, source, neighbor_centered ** 2)
        variances_i = variances_i / covariance_denominator.unsqueeze(1)
        variances_i[degrees < 2] = 0.0 #Sample variance is undefined for fewer than two neighbors
        variances_i = variances_i + eps

        distance_i = distance_function(mu_i, variances_i, mu, variances, diagonal = diagonal)
    else:
        identity = torch.eye(graph.num_features, dtype = attributes.dtype, device = device)

        # Compute empirical parameters (whole graph)
        sigma = ((centered.T @ weighted_centered) / global_covariance_denominator) + eps * identity #Degree-weighted empirical covariance matrix

        # Compute empirical parameters (neighborhood-level)
        sigma_i = torch.zeros((graph.num_nodes, graph.num_features, graph.num_features), dtype = attributes.dtype, device = device) #Initialize node-level covariances
        neighbor_outer_products = neighbor_centered.unsqueeze(2) * neighbor_centered.unsqueeze(1)
        sigma_i.index_add_(0, source, neighbor_outer_products)

        sigma_i = sigma_i / covariance_denominator.view(-1, 1, 1)
        sigma_i[degrees < 2] = 0.0 #Sample covariance is undefined for fewer than two neighbors
        sigma_i = sigma_i + eps * identity.unsqueeze(0)

        distance_i = distance_function(mu_i, sigma_i, mu, sigma, diagonal = diagonal)

    if aggregate:
        return (degree_weights * distance_i).sum() / total_degree.clamp(min = 1)
    else:
        return distance_i

def continuous_neighbor_informativeness(graph: Data, device: torch.device | str = "cpu", method: str = "P", aggregate: bool = True, eps: float = 1e-8, 
                                        diagonal: bool = False, distance: str = "KL") -> torch.Tensor:
    """
    Computes the neighborhood informativeness of a graph with continuous node attributes, through non-parametric or parametric methods.

    1) NON-PARAMETRIC METHOD (NP):
    rho_i = Wasserstein-1(P_i, Q) -> Node level

    P_i = Empirical distribution of the continuous node attributes among the neighbors of node i
    Q = Empirical distribution of the continuous node attributes among all nodes in the graph

    2) PARAMETRIC METHOD (P):
    rho_i = D(P_i, Q) -> Node level

    P_i = Gaussian distribution fitted to the continuous node attributes among the neighbors of node i
    Q = Gaussian distribution fitted to the continuous node attributes among all nodes in the graph

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    method : str
        Method to compute the neighborhood informativeness {"NP", "P"}.
    aggregate : bool
        Whether to aggregate the neighborhood informativeness to get a graph-level measure.
    eps : float
        Small value to avoid division by zero.
    diagonal : bool
        Whether to assume diagonal covariance matrices for the Gaussian distributions.
    distance : str
        Distance between Gaussian distributions {"KL", "hellinger"}.

    Returns:
    -------
    torch.Tensor
        Neighborhood informativeness of the graph.
    """
    if method == "NP":
        return non_parametric_neighbor_informativeness(graph, device = device, aggregate = aggregate)
    elif method == "P":
        return parametric_neighbor_informativeness(graph, device = device, aggregate = aggregate, eps = eps, diagonal = diagonal, distance = distance)

def neighborhood_informativeness(graph: Data, device: torch.device | str = "cpu",
                                 aggregate: bool = True,
                                 attributes_type: str = "hard",
                                 **kwargs) -> torch.Tensor:
    """
    Computes the neighborhood informativeness of an undirected graph depending on node attributes.

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph data.
    device : torch.device | str
        Device to perform computations on.
    aggregate : bool
        Whether to aggregate the informativeness to get a graph-level measure.
    attributes_type : str
        Type of node attributes {"hard", "soft", "continuous"}.
    **kwargs
        Additional arguments accepted by the selected informativeness
        function. Use ``level`` for hard or soft attributes, and ``method``,
        ``eps``, ``diagonal``, or ``distance`` for continuous attributes.

    Returns:
    -------
    torch.Tensor
        Neighborhood informativeness of the graph.
    """
    # Check if graph is valid
    if graph.num_nodes < 2 or graph.edge_index is None or graph.edge_index.size(1) == 0:
        return torch.tensor(torch.nan, device=device)
    if not hasattr(graph, "x") or graph.x is None:
        raise ValueError("Graph must have node attributes to compute neighbor informativeness")
    if graph.x.dim() != 2:
        raise ValueError((f"Errore shape: expected 2D [n, K], received {graph.x.dim()}D with shape {graph.x.shape}"))

    if attributes_type in ["hard", "soft"]:
        return jsd_informativeness(
            graph,
            device = device,
            aggregate = aggregate,
            attributes_type = attributes_type,
            **kwargs,
        )
    elif attributes_type == "continuous":
        return continuous_neighbor_informativeness(
            graph,
            device = device,
            aggregate = aggregate,
            **kwargs,
        )
    else:
        raise ValueError("attributes_type must be one of {'hard', 'soft', 'continuous'}")

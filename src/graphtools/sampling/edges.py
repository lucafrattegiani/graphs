#Mathematics and ML
import torch

#--------------------------------------------------
#GRAPHON FUNCTIONS
#--------------------------------------------------

#Ring structure
def ring_graphon(x: torch.Tensor, y: torch.Tensor, sigma: float = 0.08) -> torch.Tensor:
    """
    Ring-like graphon function. The function assigns high connection probability to pairs of nodes whose latent coordinates are close 
    with respect to the circular distance on [0, 1].

    Parameters
    ----------
    x : torch.Tensor
        Latent coordinates of the first set of nodes. It can be a scalar, vector, or any tensor broadcastable with `y`.
    y : torch.Tensor
        Latent coordinates of the second set of nodes. It must be broadcastable with `x`.
    sigma : float, default = 0.08
        Smaller values produce more local and usually sparser graphs. Larger values produce denser graphs.

    Returns
    -------
    torch.Tensor
        Connection probabilities in [0, 1].
    """
    distance = torch.abs(x - y) #Absolute distance 
    circular_distance = torch.minimum(distance, 1.0 - distance) #Circular distance
    probabilities =  torch.exp(-(circular_distance ** 2) / (2.0 * sigma ** 2)) #Connection probability

    return probabilities

#Communities (stochastic block model)
def sbm_graphon(x: torch.Tensor, y: torch.Tensor,
                block_probs: torch.Tensor | None = None) -> torch.Tensor:
    """
    Multi-community stochastic block model graphon.

    Parameters
    ----------
    x, y:
        Latent coordinates in [0, 1].
    block_probs : torch.Tensor | None, default = None
        Matrix of shape [K, K], where block_probs[a, b] is the connection
        probability between block a and block b.

    Returns
    -------
    torch.Tensor
        Connection probabilities in [0, 1].
    """
    if block_probs is None:
        block_probs = x.new_tensor([[0.8, 0.1], [0.1, 0.8]])
    else:
        block_probs = torch.as_tensor(block_probs, dtype = x.dtype, device = x.device)

    K = block_probs.shape[0] #Extract number of blocks

    x_block = torch.clamp((x * K).long(), max = K - 1) #Assign every source node to a block according to its latent coordinate
    y_block = torch.clamp((y * K).long(), max = K - 1) #Assign every ending node to a block according to its latent coordinate

    #Return connection probabilities based on the block membership of source and ending node
    probabilities = block_probs[x_block, y_block]

    return probabilities

#Bipartite
def bipartite_graphon(x: torch.Tensor, y: torch.Tensor, p_between: float = 0.9, p_within: float = 0.05) -> torch.Tensor:
    """
    Bipartite-like graphon. The interval [0, 1] is split into two parts. Nodes in different parts connect with high probability, while nodes in the same part 
    connect with low probability.

    Parameters
    ----------
    x : torch.Tensor
        Latent coordinates of the first set of nodes.
    y : torch.Tensor
        Latent coordinates of the second set of nodes.
    p_between : float, default=0.8
        Connection probability between the two latent parts.
    p_within : float, default=0.05
        Connection probability within each latent part.

    Returns
    -------
    torch.Tensor
        Connection probabilities in [0, 1].
    """
    same_left = (x < 0.5) & (y < 0.5) #(source, end) couples belonging to the left side
    same_right = (x >= 0.5) & (y >= 0.5) #(source, end) couples belonging to the right side
    same_block = same_left | same_right #(source, end) couples belonging to the same side (boolean 'o' operation)

    #Connection probabilities according to side membership (low for same side, high for different side)
    probabilities = torch.where(
        same_block, #Check when nodes belong to the same side
        torch.as_tensor(p_within, device = x.device), #Low probability to same side
        torch.as_tensor(p_between, device = x.device), #High probability to different side
    )

    return probabilities

#Random graphon
def random_graphon(x: torch.Tensor, y: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    Random graphon, all connections happen with same probability p, ignoring any possible structure.

    Parameters
    ----------
    x, y:
        Latent coordinates in [0, 1].
    p:
        Connection probability.

    Returns
    -------
    torch.Tensor
        Connection probabilities in [0, 1].
    """
    probabilities = torch.full_like(x, p)
    return probabilities

#--------------------------------------------------
#SAMPLING CONNECTIONS
#--------------------------------------------------

def sample_edges(n: int, structure: str = "random", ordered: bool = True,
                 device: torch.device | str = "cpu", **kwargs) -> torch.Tensor:
    """
    Sample undirected edges given a specified connection structure.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    structure : str, default = "random"
        Graphon structure. It can be ``"ring"``, ``"random"``, ``"sbm"``,
        or ``"bipartite"``.
    ordered : bool, default = True
        If True, sort the latent positions so that node indices follow their
        order on the latent interval.
    device : torch.device | str, default = "cpu"
        Device on which to sample latent coordinates and edges.
    **kwargs
        Additional arguments accepted by the selected graphon function.

    Returns
    -------
    torch.Tensor
        Edge indices with shape ``[2, num_edges]``. Each undirected edge is
        stored in both directions and self-loops are excluded.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be positive")
    if not isinstance(ordered, bool):
        raise TypeError("ordered must be a boolean")

    graphon_functions = {
        "ring": ring_graphon,
        "random": random_graphon,
        "sbm": sbm_graphon,
        "bipartite": bipartite_graphon,
    }
    if structure not in graphon_functions:
        supported_structures = ", ".join(graphon_functions)
        raise ValueError(
            f"Unsupported structure {structure!r}. "
            f"Choose one of: {supported_structures}."
        )

    device = torch.device(device)
    z = torch.rand(n, device = device) #Latent node positions
    if ordered:
        z = torch.sort(z).values #Node 0 has the smallest latent position

    #Generate every possible undirected edge without self-loops
    start, end = torch.triu_indices(n, n, offset = 1, device = device)
    probabilities = graphon_functions[structure](z[start], z[end], **kwargs)

    #Sample and retain the existing edges
    links = torch.rand_like(probabilities) < probabilities
    edge_start = start[links]
    edge_end = end[links]

    #Store both directions of each undirected edge
    edge_index = torch.cat(
        [
            torch.stack([edge_start, edge_end]),
            torch.stack([edge_end, edge_start]),
        ],
        dim = 1,
    )

    return edge_index

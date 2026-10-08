#Torch data and computations
import torch
import ot
from torch_geometric.data import Data

from ..utils.matrices import adjacency_matrix, laplacian_matrix, inverse_shortest_path_matrix, jaccard_similarity_matrix
from ..utils.validity import _graph_device

def fused_gromov_wasserstein_distance(graph_1: Data, graph_2: Data, alpha: float = 0.5, structure: str = "adjacency") -> tuple[torch.Tensor, dict]:
    """
    Computes the 2-Fused Gromov-Wasserstein distance between two graphs:

    FGW = min_T (1 - alpha) * sum_ij M_ij^2 T_ij
          + alpha * sum_ijkl (C1_ik - C2_jl)^2 T_ij T_kl

    M_ij -> Distance between i-th feature of graph_1 and j-th feature of graph_2.
    T -> Coupling matrix between nodes of graph_1 and graph_2.
    C_1, C_2 -> Structural similarity matrices of graph_1 and graph_2, respectively

    Parameters:
    ----------
    graph_1, graph_2 : torch_geometric.data.Data
        Graphs with possibly different sizes. Optional h contains node
        probabilities summing to one; optional C provides the structural matrix.
    alpha : float, default = 0.5
        Structural weight in [0, 1]. At 0, only features contribute; at 1,
        only structure contributes.
    structure : str, default = "adjacency"
        Matrix used when ``graph.C`` is absent or None. One of {"adjacency", "laplacian", "shortest_path", "jaccard"}.

    Returns:
    -------
    tuple[torch.Tensor, dict]
        Scalar FGW objective and a dictionary containing the transport plan
        ``T``, the weighted feature term ``w_term`` and the weighted
        structural term ``gw_term``.
    """
    graphs = [graph_1, graph_2]
    device = _graph_device(graph_1)

    # Extract node feature costs (squared Euclidean distance)
    if alpha < 1:
        features = [graph.x.to(device = device, dtype = torch.float32) for graph in graphs]
        M = torch.cdist(features[0], features[1], p = 2).square()
    else:
        M = torch.zeros((graph_1.num_nodes, graph_2.num_nodes), device = device)

    # Extract graph histograms, using uniform node weights by default
    hists = []
    for graph in graphs:
        h = getattr(graph, "h", None)
        if h is None:
            h = M.new_full((graph.num_nodes,), 1 / graph.num_nodes)
        hists.append(h.to(M))

    # Extract graph structural matrices, using graph.C when provided
    structure_functions = {
        "adjacency": adjacency_matrix,
        "laplacian": laplacian_matrix,
        "shortest_path": inverse_shortest_path_matrix,
        "jaccard": jaccard_similarity_matrix,
    }
    C = []
    for graph in graphs:
        matrix = getattr(graph, "C", None)
        if matrix is None:
            kwargs = {"normalized": False} if structure in ("adjacency", "laplacian") else {}
            matrix = structure_functions[structure](graph, device = device, sparse = False, **kwargs)
        if matrix.layout != torch.strided:
            matrix = matrix.to_dense()
        C.append(matrix.to(M))

    # Compute fused Gromov-Wasserstein distance
    distance, info = ot.gromov.fused_gromov_wasserstein2(
        M = M,
        C1 = C[0],
        C2 = C[1],
        p = hists[0],
        q = hists[1],
        loss_fun = "square_loss",
        symmetric = True,
        alpha = alpha,
        max_iter = 1000,
        tol_rel = 1e-9,
        tol_abs = 1e-9,
        log = True,
    )

    # Select and rename relevant information
    to_keep = ["T", "lin_loss", "quad_loss"]
    new_names = ["T", "w_term", "gw_term"]
    info = {new_name: info[key] for key, new_name in zip(to_keep, new_names)}

    return distance, info

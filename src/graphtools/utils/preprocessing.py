"""Graph tensor preparation shared by optimal transport functions."""

from collections.abc import Sequence

import torch
from torch_geometric.data import Data

from ..validity import _graph_device, _floating_dtype


def prepare_graphs(graph: Data, alpha: float = 0.5, *,
                   device: torch.device | str | None = None,
                   dtype: torch.dtype | None = None) -> dict:
    """
    Prepares one graph for FGW, restricting its tensors to positive-mass nodes.

    Parameters:
    ----------
    graph : torch_geometric.data.Data
        Graph with a positive node count. For alpha > 0, provide a finite,
        symmetric C of shape (n, n), dense or sparse, and optionally h of
        shape (n,) with finite nonnegative masses summing to one. For alpha
        < 1, provide finite node features x of shape (n, d).
        C must already describe the original graph: this function selects
        its submatrix and never recomputes structure after removing nodes.
    alpha : float, default = 0.5
        Structural weight in [0, 1]. At zero, ignore C and h, use uniform
        masses and a zero structural placeholder, and keep every node.
        At one, x is not read. Absent h uses uniform masses.
    device : torch.device | str | None, default = None
        Device for all returned tensors. None follows graph.edge_index,
        or CPU when absent. The input graph is never moved or modified.
    dtype : torch.dtype | None, default = None
        Floating dtype for C, h and x. None promotes the floating dtypes of
        the consumed attributes, with a minimum of float32. Explicit dtype
        allows callers to use a common precision across several graphs.

    Returns:
    -------
    dict
        C : dense structural submatrix of shape (s, s).
        h : positive node masses of shape (s,), without renormalization.
        x : feature matrix of shape (s, d), or None when alpha = 1.
        active_nodes : long tensor of original indices retained, in order.
        removed_nodes : long tensor of original indices with zero mass.
        num_nodes : original node count n, before filtering.
        Here s is the number of retained nodes. Only exact zero masses are
        removed; there is no threshold for small positive masses.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1].")
    n = graph.num_nodes
    if n is None or n < 1:
        raise ValueError("The graph must have a positive node count.")
    device = _graph_device(graph, device)
    attributes = ("x",) if alpha == 0 else (("C", "h", "x") if alpha < 1 else ("C", "h"))
    if dtype is None:
        dtype = torch.promote_types(
            _floating_dtype(*(getattr(graph, name, None) for name in attributes)),
            torch.float32,
        )
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point.")

    # Validate graph node histogram
    h = getattr(graph, "h", None) if alpha > 0 else None
    if h is None:
        h = torch.full((n,), 1 / n, device=device, dtype=dtype)
    else:
        h = h.to(device=device, dtype=dtype)
    if (h.shape != (n,) or not torch.isfinite(h).all() or (h < 0).any()
            or not torch.isclose(h.sum(), h.new_tensor(1.))):
        raise ValueError("h must contain n finite nonnegative masses summing to one.")
    
    # Select active nodes having positive probability mass
    active_nodes = torch.nonzero(h > 0, as_tuple=True)[0]
    removed_nodes = torch.nonzero(h == 0, as_tuple=True)[0]

    # Extract strctural and feature tensors
    if alpha == 0:
        C = torch.zeros((n, n), device=device, dtype=dtype)
    else:
        C = getattr(graph, "C", None)
        if C is None or C.shape != (n, n):
            raise ValueError("For alpha > 0, provide C with shape (n, n).")
        C = C.to(device=device, dtype=dtype)
        if C.layout != torch.strided:
            C = C.to_dense()
        if not torch.isfinite(C).all() or not torch.allclose(C, C.T):
            raise ValueError("C must be finite and symmetric.")

    x = None
    if alpha < 1:
        x = graph.x
        if x is None or x.ndim != 2 or x.shape[0] != n:
            raise ValueError("For alpha < 1, provide x with shape (n, d).")
        x = x.to(device=device, dtype=dtype)
        if not torch.isfinite(x).all():
            raise ValueError("x must be finite.")
        x = x[active_nodes]

    # Filter out non-active nodes and store them for later reconstruction
    return {"C": C[active_nodes][:, active_nodes], "h": h[active_nodes], "x": x,
            "active_nodes": active_nodes, "removed_nodes": removed_nodes,
            "num_nodes": n}


def prepare_batches(batch: Sequence[Data], alpha: float, *,
                    device: torch.device | str, dtype: torch.dtype) -> dict[int, dict]:
    """
    Prepares graphs and groups them by their number of positive-mass nodes.

    Parameters:
    ----------
    batch : Sequence[torch_geometric.data.Data]
        Nonempty graph sequence. Original node counts may differ. Attributes
        are validated and zero-mass nodes filtered by prepare_graphs.
    alpha : float
        FGW structural weight in [0, 1]. At zero, h is ignored and all
        original nodes retain uniform mass. At one, features are not used.
    device : torch.device | str
        Target device for all tensors, without modifying the input graphs.
    dtype : torch.dtype
        Common floating precision chosen by the caller for both batches.

    Returns:
    -------
    dict[int, dict]
        Maps each support size s to a group containing stacked C (B, s, s),
        h (B, s), x (B, s, d) or None, and indices (B,) giving the original
        graph positions in the batch. B is the number of graphs in the group.
        Lists active_nodes, removed_nodes and num_nodes preserve each graph's
        node mapping and original size in the same order as indices.
        No padding is added. Feature width must agree across all groups.
    """
    if len(batch) == 0:
        raise ValueError("The batch must be nonempty.")

    # Group graphs in sub-batches of equal graph sizes
    groups = {}
    feature_width = None # Test on eventually mismatching feature dimensions
    for index, graph in enumerate(batch):
        prepared = prepare_graphs(graph, alpha, device=device, dtype=dtype)
        if alpha < 1: # Check feature dimension consistency across all graphs
            width = prepared["x"].shape[1]
            if feature_width is not None and width != feature_width:
                raise ValueError("Node features must have a common width.")
            feature_width = width

        # Extract the size of the current graph and check if it falls into an existing group
        size = prepared["h"].numel()
        if size not in groups:
            groups[size] = {"indices": [], "graphs": []}
        groups[size]["indices"].append(index)
        groups[size]["graphs"].append(prepared)

    # Stack the tensors of each group for compatibility with POT solver
    for size, group in groups.items():
        graphs = group.pop("graphs")
        group["indices"] = torch.tensor(group["indices"], device=device, dtype=torch.long)
        group["C"] = torch.stack([graph["C"] for graph in graphs])
        group["h"] = torch.stack([graph["h"] for graph in graphs])
        group["x"] = torch.stack([graph["x"] for graph in graphs]) if alpha < 1 else None
        for name in ("active_nodes", "removed_nodes", "num_nodes"):
            group[name] = [graph[name] for graph in graphs]
    return groups

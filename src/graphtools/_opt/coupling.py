from collections.abc import Sequence
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data
import torch
import ot

from ..validity import _graph_device, _floating_dtype
from ..utils.preprocessing import prepare_batches


@torch.no_grad()
def hungarian_coupling(
    batch_1: Sequence[Data],
    batch_2: Sequence[Data],
    alpha: float = 0.5,
    *,
    device: torch.device | str | None = None,
    pair_batch_size: int = 32,
    reg: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-5,
    max_iter_inner: int = 100,
    tol_inner: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """
    Computes an optimal uniform coupling between two batches of graphs:

    C_ij = FGW_alpha(batch_1[i], batch_2[j])
    W = argmin_W sum_ij W_ij C_ij
        subject to W >= 0 and row/column sums equal to 1/K.

    C -> Cross-batch graph cost matrix, computed with POT's batched FGW solver.
    W -> Graph-level coupling obtained by a one-to-one Linear assignment.
    K -> Number of graphs in each batch.

    Parameters:
    ----------
    batch_1, batch_2 : Sequence[torch_geometric.data.Data]
        Nonempty lists or tuples containing the same number K of graphs.
        Node counts must be positive but may vary within and between batches.
        Zero-mass nodes are filtered locally, then graphs are grouped by
        support size for computation without padding. For alpha > 0, provide a finite,
        symmetric structural matrix ``C`` of shape (n, n), dense or sparse.
        For alpha < 1, ``x`` of shape (n, d) is required, with the same feature
        dimension d for all graphs. Optional ``h`` of shape (n,) contains
        finite, nonnegative node masses summing to one; absent h uses uniform
        node masses. At alpha = 0, only x is used: C and h are ignored,
        and node masses are uniform. Graph-level masses are always 1/K.
        Input graphs and their tensors are not modified.
    alpha : float, default = 0.5
        Structural weight in [0, 1]. At 0, only squared Euclidean distances
        between node features with uniform node masses contribute;
        at 1, only structure contributes
        and node features are not required.
    device : torch.device | str | None, default = None
        Device for FGW computation and all returned tensors. If None, follow
        batch_1[0].edge_index (CPU if absent). Linear assignment runs on
        CPU, then its result is transferred to this device. An explicitly
        requested unavailable device raises an error.
    pair_batch_size : int, default = 32
        Maximum number of graph pairs solved simultaneously. Smaller values
        reduce intermediate memory usage. All K*K cross-batch pairs are
        evaluated; no within-batch distances are computed.
    reg : float, default = 0.01
        Finite positive coefficient of the KL proximal term in the FGW
        solver. Controls each optimization step; it is not an entropy penalty
        added to the returned FGW cost.
    max_iter : int, default = 50
        Maximum number of outer proximal iterations for each block of pairs.
    tol : float, default = 1e-5
        Nonnegative outer stopping tolerance on the maximum, across pairs,
        of the mean absolute change in the node transport plan.
    max_iter_inner : int, default = 100
        Maximum number of inner Bregman projection iterations per outer step.
        These projections enforce the node-mass constraints.
    tol_inner : float, default = 1e-6
        Nonnegative inner stopping tolerance on the maximum, across pairs,
        of the L2 row-marginal residual, checked periodically by POT.

    Returns:
    -------
    tuple[torch.Tensor, dict]
        Graph-level coupling ``W`` of shape (K, K), with exactly one entry
        1/K per row and column, and a dictionary containing:
        ``C`` : tensor of shape (K, K), with the FGW cost for each cross pair.
        ``assignment`` : long tensor of shape (K,), giving the target index
            in batch_2 assigned to each graph in batch_1.
        ``cost`` : scalar tensor equal to sum(W * C), the average matched cost.
        All tensors use the selected device. Floating input dtypes are
        promoted to a common dtype, with a minimum precision of float32.

    Notes:
    -----
    FGW costs are numerical approximations and may differ from the package's
    CPU conditional-gradient solver. Linear assignment is optimal for the
    computed cost matrix; global optimality of each FGW solve is not guaranteed.
    Original graph indices are preserved across groups. Only zero-mass nodes
    are excluded from the solver; structural costs are not recomputed.
    Gradients are disabled and node-level transport plans are not retained.
    """
    # 1. Check the batch contract and select one device/dtype for all pairs.
    if len(batch_1) == 0 or len(batch_1) != len(batch_2):
        raise ValueError("Batches must be nonempty and have the same length.")
    if not 0 <= alpha <= 1 or not 0 < reg < float("inf"):
        raise ValueError("Require alpha in [0, 1] and finite reg > 0.")
    for name, value in (("pair_batch_size", pair_batch_size),
                        ("max_iter", max_iter), ("max_iter_inner", max_iter_inner)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not 0 <= tol < float("inf") or not 0 <= tol_inner < float("inf"):
        raise ValueError("Tolerances must be finite and nonnegative.")

    device = _graph_device(batch_1[0], device)
    # Select only attributes used by the chosen objective.
    attributes = ("x",) if alpha == 0 else (("C", "h", "x") if alpha < 1 else ("C", "h"))
    dtype = _floating_dtype(*(getattr(g, name, None)
                              for batch in (batch_1, batch_2)
                              for g in batch for name in attributes))
    dtype = torch.promote_types(dtype, torch.float32)
    K = len(batch_1) # batch size

    groups_1 = prepare_batches(batch_1, alpha, device=device, dtype=dtype)
    groups_2 = prepare_batches(batch_2, alpha, device=device, dtype=dtype)
    if alpha < 1:
        x1 = next(iter(groups_1.values()))["x"]
        x2 = next(iter(groups_2.values()))["x"]
        if x1.shape[2] != x2.shape[2]:
            raise ValueError("The two batches must have the same feature width.")

    # Compute every cross-batch cost, grouping only the numerical FGW solves.
    # Each pair of groups has fixed node counts, even for heterogeneous inputs.
    C = torch.empty((K, K), device=device, dtype=dtype)
    for group_1 in groups_1.values():
        for group_2 in groups_2.values():
            C1, h1, x1 = group_1["C"], group_1["h"], group_1["x"]
            C2, h2, x2 = group_2["C"], group_2["h"], group_2["x"]
            count_1, count_2 = C1.shape[0], C2.shape[0]
            # Flatten the rectangular group-pair block, then solve in chunks
            # to limit memory. There is no local matching within these blocks.
            for start in range(0, count_1 * count_2, pair_batch_size):
                stop = min(start + pair_batch_size, count_1 * count_2)
                pairs = torch.arange(start, stop, device=device)
                i, j = pairs // count_2, pairs % count_2
                M = (torch.cdist(x1[i], x2[j]).square() if alpha < 1
                     else C1.new_zeros((stop - start, C1.shape[1], C2.shape[1])))
                result = ot.batch.solve_gromov_batch(
                    Ca=C1[i], Cb=C2[j], a=h1[i], b=h2[j], M=M,
                    alpha=alpha, loss="sqeuclidean", symmetric=True, reg=reg,
                    max_iter=max_iter, tol=tol,
                    max_iter_inner=max_iter_inner, tol_inner=tol_inner, grad="detach",
                )
                # Restore original batch positions, rather than group order.
                C[group_1["indices"][i], group_2["indices"][j]] = result.value

    # Find one global optimal coupling after filling all K*K costs.
    if not torch.isfinite(C).all():
        raise RuntimeError("FGW produced nonfinite results; check inputs and solver settings.")
    rows, columns = linear_sum_assignment(C.cpu().numpy()) # Solve the optimization problem using the 1:1 assignment solution
    rows = torch.as_tensor(rows, device=device) # Indexes of the graphs from the first batch
    columns = torch.as_tensor(columns, device=device) # Corresponding 1:1 indexes of the graphs from the second batch
    assignment = torch.empty(K, dtype=torch.long, device=device)
    assignment[rows] = columns # For every graph in the first batch (rows), assign a single graph in the second batch (columns)
    W = torch.zeros_like(C)
    W[rows, columns] = 1 / K # Every entry [rows[i], columns[i]] corresponding to an existing assignment is set to 1/K to satisfy the marginal constraint
    return W, {"C": C, "assignment": assignment, "cost": (W * C).sum()}

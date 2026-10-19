"""Graph tensor preparation shared by optimal transport functions."""

from collections.abc import Sequence

import torch
from torch_geometric.data import Data


def prepare_batches(batch: Sequence[Data], alpha: float, *,
                    device: torch.device | str, dtype: torch.dtype
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Prepares one batch of graphs for batched fused Gromov-Wasserstein solves.

    Parameters:
    ----------
    batch : Sequence[torch_geometric.data.Data]
        Nonempty list or tuple of graphs with the same positive node count n.
        For alpha > 0, C must be a finite symmetric (n, n) matrix, dense or
        sparse. Optional h must contain n nonnegative masses summing to one.
        For alpha < 1, x must be finite with shape (n, d), with a common d.
    alpha : float
        FGW structural weight in [0, 1], validated by the caller. At zero,
        only x is read: zero structural placeholders and uniform masses are
        generated for POT. At one, x is not read and features returns None.
    device : torch.device | str
        Target device for the returned tensors; input graphs are unchanged.
    dtype : torch.dtype
        Common floating dtype selected by the caller for both batches.

    Returns:
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        Dense structural matrices (K, n, n), node masses (K, n), and features
        (K, n, d) or None. K is the number of graphs in this batch. Absent h
        uses uniform masses; supplied h is validated, not renormalized.
        Sparse structural matrices are densified. Input tensors are not
        modified; all results use the supplied device and dtype.
    """
    K = len(batch)
    if K == 0:
        raise ValueError("The batch must be nonempty.")
    n = batch[0].num_nodes
    if n is None or n < 1 or any(g.num_nodes != n for g in batch):
        raise ValueError("Each batch must have a fixed positive node count.")
    structures, masses, features = [], [], []
    for graph in batch:
        if alpha == 0:
            # POT still requires structural arrays. Zero placeholders keep
            # the same solver interface without reading graph.C or graph.h.
            matrix = torch.zeros((n, n), device=device, dtype=dtype)
            h = None
        else:
            matrix = getattr(graph, "C", None)
            if matrix is None or matrix.shape != (n, n):
                raise ValueError("For alpha > 0, provide C with shape (n, n).")
            matrix = matrix.to(device=device, dtype=dtype)
            if matrix.layout != torch.strided:
                matrix = matrix.to_dense()
            h = getattr(graph, "h", None)
        structures.append(matrix)
        h = matrix.new_full((n,), 1 / n) if h is None else h.to(matrix)
        if h.shape != (n,):
            raise ValueError("Each h must have shape (n,).")
        masses.append(h)
        if alpha < 1:
            x = graph.x
            if x is None or x.ndim != 2 or x.shape[0] != n:
                raise ValueError("For alpha < 1, each x must have shape (n, d).")
            features.append(x.to(matrix))

    structures, masses = torch.stack(structures), torch.stack(masses)
    if not torch.isfinite(structures).all() or not torch.allclose(
        structures, structures.transpose(1, 2)
    ):
        raise ValueError("Structural matrices must be finite and symmetric.")
    if (not torch.isfinite(masses).all() or (masses < 0).any()
            or not torch.allclose(masses.sum(1), masses.new_ones(K))):
        raise ValueError("Node masses must be finite, nonnegative and sum to one.")
    if features and any(x.shape[1] != features[0].shape[1] for x in features):
        raise ValueError("Node features must have a common width.")
    features = torch.stack(features) if features else None
    if features is not None and not torch.isfinite(features).all():
        raise ValueError("Node features must be finite.")
    return structures, masses, features

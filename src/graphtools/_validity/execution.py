"""Execution placement, floating-point precision and input validation."""

import torch


def _graph_device(graph, device: torch.device | str | None = None,
                  *, attributes: tuple[str, ...] = ("edge_index",)) -> torch.device:
    """Resolve device placement using only the tensors consumed by the operation.

    An explicit device permits local transfers. Otherwise, consumed tensors
    must agree on their device. This helper never transfers or mutates data.
    """
    if device is not None:
        return torch.device(device)

    devices = set()
    for name in attributes:
        value = getattr(graph, name, None)
        if isinstance(value, torch.Tensor):
            devices.add(value.device)
    if len(devices) > 1:
        raise ValueError(
            f"Graph attributes {attributes} must be on the same device; "
            "move them explicitly or pass device to permit local transfers."
        )
    return next(iter(devices), torch.device("cpu"))


def _floating_dtype(*tensors: torch.Tensor | None) -> torch.dtype:
    """Promote floating input dtypes; use float32 for integer-only inputs."""
    dtype = None
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            dtype = tensor.dtype if dtype is None else torch.promote_types(dtype, tensor.dtype)
    return torch.float32 if dtype is None else dtype


def _validate_size(value: int, name: str) -> None:
    """Validate a positive integer size parameter."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _class_probabilities(K: int, p: list[float] | torch.Tensor,
                         device: torch.device | str) -> torch.Tensor:
    """Create and validate a probability vector for K classes."""
    probabilities = torch.as_tensor(p, dtype = torch.float, device = device)

    if probabilities.numel() == 0:
        return torch.full((K,), 1.0 / K, device = device)
    if probabilities.ndim != 1 or probabilities.numel() != K:
        raise ValueError(f"p must contain exactly {K} probabilities")
    if not torch.isfinite(probabilities).all():
        raise ValueError("p must contain only finite probabilities")
    if (probabilities < 0).any():
        raise ValueError("p probabilities must be non-negative")
    if not torch.isclose(
        probabilities.sum(),
        torch.tensor(1.0, device = device),
    ):
        raise ValueError("p probabilities must sum to one")

    return probabilities

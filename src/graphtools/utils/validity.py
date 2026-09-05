#Torch data and computations
import torch


def _graph_device(graph, device: torch.device | str | None = None) -> torch.device:
    """Return an explicit device or infer it from a graph tensor."""
    if device is not None:
        return torch.device(device)

    for attribute in ("edge_index", "x", "pos"):
        value = getattr(graph, attribute, None)
        if isinstance(value, torch.Tensor):
            return value.device

    return torch.device("cpu")


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

"""Shared categorical palettes and probability-weighted RGB colors."""
import matplotlib as mpl
import torch

DEFAULT_PALETTE = "tab20"


def set_colors(K: int, palette: str | None = None) -> torch.Tensor:
    """Return K pure RGB colors as a CPU float32 tensor of shape [K, 3].

    palette names a categorical Matplotlib palette (default: DEFAULT_PALETTE).
    Colors follow its fixed order, independently of K. Requests exceeding
    the palette size raise an error rather than silently repeating colors.
    """
    if not isinstance(K, int) or isinstance(K, bool) or K < 1:
        raise ValueError("K must be a positive integer")
    cmap = mpl.colormaps[palette or DEFAULT_PALETTE]
    if not hasattr(cmap, "colors"):
        raise ValueError("Choose a categorical palette with a fixed color list")
    if K > len(cmap.colors):
        raise ValueError(f"Palette {cmap.name!r} contains only {len(cmap.colors)} colors")
    return torch.tensor([mpl.colors.to_rgb(c) for c in cmap.colors[:K]], dtype=torch.float32)


def color_labels(labels: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    """Mix pure colors [K, 3] using probability rows labels [n, K].

    Return RGB colors [n, 3] on the labels' device, without changing inputs.
    One-hot rows select pure colors. Invalid probabilities raise an error;
    rows are never silently normalized. Integer one-hot matrices are accepted.
    """
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape[1] < 1:
        raise ValueError("labels must be a probability matrix with shape [n, K]")
    if labels.is_complex():
        raise ValueError("labels must contain real probabilities")
    weights = labels.to(dtype=torch.float64 if labels.dtype == torch.float64 else torch.float32)
    if (not torch.isfinite(weights).all() or (weights < 0).any()
            or not torch.allclose(weights.sum(-1), weights.new_ones(weights.shape[0]))):
        raise ValueError("labels must contain finite nonnegative rows summing to one")
    if colors.shape != (labels.shape[1], 3) or not torch.isfinite(colors).all() or ((colors < 0) | (colors > 1)).any():
        raise ValueError("colors must have shape [K, 3] with RGB values in [0, 1]")
    return (weights @ colors.to(weights)).clamp(0, 1)

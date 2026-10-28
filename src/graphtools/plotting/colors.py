"""Shared categorical palettes and entropy-shaded category colors."""
from math import log

import matplotlib as mpl
import torch

DEFAULT_PALETTE = "tab10"
UNCERTAINTY_COLOR = (0.85, 0.85, 0.85)


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
    # Matplotlib needs host colors, even when Torch's default device is CUDA.
    return torch.tensor([mpl.colors.to_rgb(c) for c in cmap.colors[:K]],
                        dtype=torch.float32, device="cpu")


def color_labels(labels: torch.Tensor, colors: torch.Tensor,
                 mode: str = "standard", label_index: int = 0) -> torch.Tensor:
    """Shade the dominant category color by normalized entropy of labels [n, K].

    Return RGB colors [n, 3] on the labels' device, without changing inputs.
    mode="heatmap" instead blends white with the selected category color
    using its probability, on a fixed 0–1 scale. label_index selects its column.

    colors contains K pure RGB colors with shape [K, 3]. The result is
    (1 - u) * colors[argmax(labels)] + u * UNCERTAINTY_COLOR,
    where u = H(labels) / log(K). One-hot rows keep their pure color;
    uniform rows become neutral gray. For K = 1, uncertainty is zero.
    Tied maxima select the first category, following torch.argmax.
    Invalid probabilities raise an error;
    rows are never silently normalized. Integer one-hot matrices are accepted.
    """
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape[1] < 1:
        raise ValueError("labels must be a probability matrix with shape [n, K]")
    weights = labels.to(dtype=torch.float64 if labels.dtype == torch.float64 else torch.float32)
    if (not torch.isfinite(weights).all() or (weights < 0).any()
            or not torch.allclose(weights.sum(-1), weights.new_ones(weights.shape[0]))):
        raise ValueError("labels must contain finite nonnegative rows summing to one")
    if colors.shape != (labels.shape[1], 3) or not torch.isfinite(colors).all() or ((colors < 0) | (colors > 1)).any():
        raise ValueError("colors must have shape [K, 3] with RGB values in [0, 1]")
    if mode not in ("standard", "heatmap"):
        raise ValueError('mode must be "standard" or "heatmap"')
    if mode == "heatmap":
        if not isinstance(label_index, int) or isinstance(label_index, bool) or not 0 <= label_index < weights.shape[1]:
            raise ValueError("label_index is outside the available categories")
        return heatmap_colors(weights, colors.to(weights), label_index)

    # Select the category first: uncertainty never mixes different category hues.
    dominant_colors = colors.to(weights)[weights.argmax(dim=-1)]
    if weights.shape[1] == 1:
        return dominant_colors

    # xlogy(0, 0) = 0 handles one-hot assignments without adding epsilon.
    entropy = -torch.special.xlogy(weights, weights).sum(dim=-1)
    uncertainty = (entropy / log(weights.shape[1])).clamp(0, 1)[:, None]
    neutral = weights.new_tensor(UNCERTAINTY_COLOR)
    return ((1 - uncertainty) * dominant_colors + uncertainty * neutral).clamp(0, 1)


def heatmap_colors(labels: torch.Tensor, colors: torch.Tensor, index: int) -> torch.Tensor:
    """Mix white and one pure color using already validated probabilities."""
    probability = labels[:, index:index + 1]
    return ((1 - probability) + probability * colors[index]).clamp(0, 1)

#Torch data and computations
import numpy as np
import torch
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from torch_geometric.data import Data

#Plotting
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from ..utils.matrices import adjacency_matrix

_VORONOI_FACE_COLOR = to_rgba("lightsteelblue", alpha = 0.65)
_VORONOI_EDGE_COLOR = to_rgba("tab:blue", alpha = 0.8)
_VORONOI_EDGE_ALPHA = 0.8
_VORONOI_EDGE_WIDTH = 1.0

def rescale(positions: torch.Tensor, lim_inf: float = -1.0, lim_sup: float = 1.0) -> torch.Tensor:
    """
    Rescale the spatial coordinates to stay in [lim_inf, lim_sup].

    Parameters
    ----------
    positions : torch.Tensor
        Spatial coordinates of shape [n, 2].
    lim_inf : float, default = -1.0
        Lower limit of the target coordinate range.
    lim_sup : float, default = 1.0
        Upper limit of the target coordinate range.

    Returns
    -------
    torch.Tensor
        Rescaled spatial coordinates in [lim_inf, lim_sup].
    """
    min_pos = positions.amin(dim = 0) #Lowest coordinates (x, y)
    max_pos = positions.amax(dim = 0) #Highest coordinates (x, y)
    range_pos = max_pos - min_pos #Range of coordinates (x, y)
    range_pos = torch.where(
        range_pos > 0,
        range_pos,
        torch.ones_like(range_pos),
    )

    #Rescale coordinates to [lim_inf, lim_sup]
    positions = (lim_sup - lim_inf) * (positions - min_pos) / range_pos + lim_inf

    return positions


def plot_voronoi(voronoi: Voronoi, title: str = "Voronoi tessellation",
                 ax: Axes | None = None) -> None:
    """Plot Voronoi cells without drawing their generating points.

    Parameters
    ----------
    voronoi : scipy.spatial.Voronoi
        Voronoi tessellation to plot.
    title : str, default = "Voronoi tessellation"
        Title displayed above the plot.
    ax : matplotlib.axes.Axes | None, default = None
        Axes on which to draw. If None, a new figure is created and displayed.

    Returns
    -------
    None
        The function draws the tessellation and does not return a value. The
        figure is displayed only when ``ax`` is None.
    """
    if not isinstance(voronoi, Voronoi):
        raise TypeError("voronoi must be a scipy.spatial.Voronoi object")
    if not isinstance(title, str):
        raise TypeError("title must be a string")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize = (7, 7))
    else:
        fig = ax.figure

    # Standard Voronoi cells cover the whole plane. Coloring the axes gives
    # both finite and infinite cells the same fill used by epsilon-Voronoi.
    ax.set_facecolor(_VORONOI_FACE_COLOR)

    voronoi_plot_2d(
        voronoi,
        ax = ax,
        show_points = False,
        show_vertices = False,
        line_colors = _VORONOI_EDGE_COLOR,
        line_width = _VORONOI_EDGE_WIDTH,
        line_alpha = _VORONOI_EDGE_ALPHA,
    )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable = "box")
    ax.axis("off")

    if standalone:
        fig.tight_layout()
        plt.show()


def plot_epsilon_voronoi(cells: list[Polygon],
                         title: str = "Epsilon-Voronoi tessellation",
                         ax: Axes | None = None) -> None:
    """Plot epsilon-Voronoi cells without drawing their generating points.

    Parameters
    ----------
    cells : list[shapely.geometry.Polygon]
        Epsilon-Voronoi cells to plot.
    title : str, default = "Epsilon-Voronoi tessellation"
        Title displayed above the plot.
    ax : matplotlib.axes.Axes | None, default = None
        Axes on which to draw. If None, a new figure is created and displayed.

    Returns
    -------
    None
        The function draws the tessellation and does not return a value. The
        figure is displayed only when ``ax`` is None.
    """
    if not isinstance(cells, list):
        raise TypeError("cells must be a list of shapely.geometry.Polygon objects")
    if not all(isinstance(cell, Polygon) for cell in cells):
        raise TypeError("cells must contain only shapely.geometry.Polygon objects")
    if not isinstance(title, str):
        raise TypeError("title must be a string")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize = (7, 7))
    else:
        fig = ax.figure

    for cell in cells:
        if cell.is_empty:
            continue

        coordinates = np.asarray(cell.exterior.coords)
        ax.fill(
            coordinates[:, 0],
            coordinates[:, 1],
            facecolor = _VORONOI_FACE_COLOR,
            edgecolor = _VORONOI_EDGE_COLOR,
            linewidth = _VORONOI_EDGE_WIDTH,
            alpha = _VORONOI_EDGE_ALPHA,
        )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable = "box")
    ax.autoscale_view()
    ax.axis("off")

    if standalone:
        fig.tight_layout()
        plt.show()


def plot_positions(positions: torch.Tensor, edge_index: torch.Tensor | None = None,
                   directed: bool = False, title: str = "", ax: Axes | None = None,
                   labels: torch.Tensor | None = None,
                   voronoi: Voronoi | list[Polygon] | None = None) -> None:
    """Plot two-dimensional spatial positions and their optional connections.

    Parameters
    ----------
    positions : torch.Tensor
        Spatial coordinates with shape ``[num_nodes, 2]``.
    edge_index : torch.Tensor | None, default = None
        Optional connections with shape ``[2, num_edges]``. For an undirected
        graph stored in both directions, only one direction is plotted.
    directed : bool, default = False
        Whether ``edge_index`` represents a directed graph.
    title : str, default = ""
        Title displayed above the plot. If empty, the number of positions and
        connections is used.
    ax : matplotlib.axes.Axes | None, default = None
        Axes on which to draw. If None, a new figure is created and displayed.
    labels : torch.Tensor | None, default = None
        Optional integer class assignment for each position, with shape
        ``[num_nodes]`` or ``[num_nodes, 1]``.
    voronoi : scipy.spatial.Voronoi | list[shapely.geometry.Polygon] | None, default = None
        Optional standard or epsilon-Voronoi tessellation to draw behind the
        positions.

    Returns
    -------
    None
        The function draws the positions and does not return a value. The
        figure is displayed only when ``ax`` is None.
    """
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 1:
        raise ValueError("positions must have shape [num_nodes, 2] with at least one node")

    num_nodes = positions.shape[0]
    nodes = positions.detach().cpu().numpy()

    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a torch.Tensor")
        if labels.ndim == 2 and labels.shape == (num_nodes, 1):
            labels = labels.squeeze(1)
        elif labels.ndim != 1 or labels.numel() != num_nodes:
            raise ValueError("labels must have shape [num_nodes] or [num_nodes, 1]")
        if (
            labels.dtype == torch.bool
            or torch.is_floating_point(labels)
            or torch.is_complex(labels)
        ):
            raise TypeError("labels must contain integers")
        labels = labels.detach().cpu().numpy()

    if edge_index is not None and edge_index.numel() > 0:
        links = edge_index.detach().cpu().numpy()
        src_nodes = links[0]
        dst_nodes = links[1]

        if not directed:
            unique_edges = src_nodes < dst_nodes
            src_nodes = src_nodes[unique_edges]
            dst_nodes = dst_nodes[unique_edges]

        segments = np.stack(
            [
                nodes[src_nodes],
                nodes[dst_nodes],
            ],
            axis = 1,
        )
        num_edges = len(src_nodes)
    else:
        segments = np.empty((0, 2, 2))
        num_edges = 0

    edge_alpha = min(0.35, max(0.015, 800.0 / max(num_edges, 1)))
    edge_width = min(0.6, max(0.08, 200.0 / max(num_edges, 1)))
    node_size = max(35, max(3, 1000.0 / num_nodes))

    finite_nodes = nodes[np.isfinite(nodes).all(axis = 1)]
    if len(finite_nodes) == 0:
        raise ValueError("positions must contain at least one finite coordinate pair")

    coord_min = finite_nodes.min(axis = 0)
    coord_max = finite_nodes.max(axis = 0)
    coord_span = coord_max - coord_min
    fallback_span = max(float(coord_span.max()), 1.0)
    coord_pad = np.where(
        coord_span > 0,
        0.08 * coord_span,
        0.08 * fallback_span,
    )
    plot_min = coord_min - coord_pad
    plot_max = coord_max + coord_pad

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize = (7, 7))
    else:
        fig = ax.figure

    if isinstance(voronoi, Voronoi):
        plot_voronoi(voronoi, title = title, ax = ax)
    elif isinstance(voronoi, list):
        plot_epsilon_voronoi(voronoi, title = title, ax = ax)
        nonempty_cells = [cell for cell in voronoi if not cell.is_empty]
        if nonempty_cells:
            cell_bounds = np.asarray([cell.bounds for cell in nonempty_cells])
            plot_min = np.minimum(plot_min, cell_bounds[:, :2].min(axis = 0))
            plot_max = np.maximum(plot_max, cell_bounds[:, 2:].max(axis = 0))
    elif voronoi is not None:
        raise TypeError(
            "voronoi must be a scipy.spatial.Voronoi object, "
            "a list of shapely.geometry.Polygon objects, or None"
        )

    if num_edges > 0:
        ax.add_collection(
            LineCollection(
                segments,
                colors = "black",
                linewidths = edge_width,
                alpha = edge_alpha,
                zorder = 1,
            )
        )

    if labels is None:
        ax.scatter(nodes[:, 0], nodes[:, 1], s = node_size)
    else:
        unique_labels = np.unique(labels)
        color_map = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")

        for color_index, class_id in enumerate(unique_labels):
            class_nodes = labels == class_id
            ax.scatter(
                nodes[class_nodes, 0],
                nodes[class_nodes, 1],
                s = node_size,
                color = color_map(color_index % color_map.N),
                label = str(class_id),
            )

        ax.legend(title = "Labels")

    plot_title = title or f"n={num_nodes}, edges={num_edges}"
    ax.set_title(plot_title)
    ax.set_aspect("equal", adjustable = "box")
    ax.set_xlim(plot_min[0], plot_max[0])
    ax.set_ylim(plot_min[1], plot_max[1])
    ax.axis("off")
    if standalone:
        fig.tight_layout()
        plt.show()


def plot_nodes(graph: Data, title: str = "", ax: Axes | None = None,
               voronoi: bool = False) -> None:
    """Plot a graph using the spatial coordinates stored in ``graph.pos``.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        Graph to plot. Its connectivity is read from ``edge_index`` and its
        spatial coordinates from ``pos``.
    title : str, default = ""
        Title displayed above the graph. If empty, the number of nodes and
        edges is used.
    ax : matplotlib.axes.Axes | None, default = None
        Axes on which to draw the graph. If None, a new figure is created and
        displayed.
    voronoi : bool, default = False
        Whether to draw the Voronoi cells stored in ``graph.cells`` behind
        the graph.

    Returns
    -------
    None
        The function draws the graph and does not return a value. If
        ``graph.pos`` is missing, an empty plot titled
        ``"No positions detected"`` is drawn. The figure is displayed only
        when ``ax`` is None.
    """
    if not isinstance(graph, Data):
        raise TypeError("graph must be a torch_geometric.data.Data object")
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if not isinstance(voronoi, bool):
        raise TypeError("voronoi must be a boolean")

    positions = getattr(graph, "pos", None)
    if positions is None or (isinstance(positions, torch.Tensor) and positions.numel() == 0):
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize = (7, 7))
        else:
            fig = ax.figure
        ax.set_title("No positions detected")
        ax.axis("off")
        if standalone:
            fig.tight_layout()
            plt.show()
        return
    if not isinstance(positions, torch.Tensor):
        raise TypeError("graph.pos must be a torch.Tensor")

    if graph.num_nodes is None or graph.num_nodes < 1:
        raise ValueError("graph must contain at least one node")
    if positions.ndim != 2 or positions.shape != (graph.num_nodes, 2):
        raise ValueError("graph.pos must have shape [num_nodes, 2]")

    cells = None
    if voronoi:
        cells = getattr(graph, "cells", None)
        if cells is None:
            raise ValueError(
                "graph.cells must contain a standard or epsilon-Voronoi "
                "tessellation when voronoi=True"
            )

    plot_positions(
        positions,
        edge_index = getattr(graph, "edge_index", None),
        directed = graph.is_directed(),
        title = title,
        ax = ax,
        labels = getattr(graph, "labels", None),
        voronoi = cells,
    )


def plot_heatmap(matrix: torch.Tensor, title: str, figsize: tuple[float, float] = (7, 7),
                 ax: Axes | None = None) -> None:
    """
    Plots a matrix as a heatmap.

    Parameters:
    ----------
    matrix : torch.Tensor
        Two-dimensional dense or sparse COO tensor to visualize. Its rows and
        columns are interpreted as graph nodes.
    figsize : tuple[float, float]
        Figure width and height in inches.
    title : str
        Title displayed above the heatmap.
    ax : matplotlib.axes.Axes | None, default = None
        Axes on which to draw the heatmap. If None, a new figure is created
        and displayed.

    Returns:
    -------
    None
        The function draws the heatmap and does not return a value. It displays
        the figure only when ``ax`` is None.

    Notes:
    -----
    Converting a sparse matrix to dense requires memory proportional to the
    square of the number of nodes. Therefore, plotting very large graph
    matrices may require substantial memory.
    """

    dense_matrix = matrix.to_dense() if matrix.is_sparse else matrix
    matrix_values = dense_matrix.detach().cpu().numpy()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize = figsize)
    else:
        fig = ax.figure

    image = ax.imshow(matrix_values, interpolation = "nearest")
    ax.set_title(title)
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Nodes")
    fig.colorbar(image, ax = ax, fraction = 0.046, pad = 0.04)

    if standalone:
        fig.tight_layout()
        plt.show()


def plot_graph(graph: Data, device: torch.device | str = "cpu", title: str = "",
               figsize: tuple[float, float] = (15, 7),
               voronoi: bool = False) -> None:
    """Plot a graph's adjacency matrix and spatial representation side by side.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        Graph whose adjacency matrix and spatial representation are plotted.
    device : torch.device | str, default = "cpu"
        Device on which to compute the adjacency matrix.
    title : str, default = ""
        Overall figure title. If empty, the number of nodes and edges is used.
    figsize : tuple[float, float], default = (15, 7)
        Figure width and height in inches.
    voronoi : bool, default = False
        Whether to draw the Voronoi cells stored in ``graph.cells`` behind
        the spatial representation.

    Returns
    -------
    None
        The function displays a Matplotlib figure and does not return a value.
    """
    if not isinstance(graph, Data):
        raise TypeError("graph must be a torch_geometric.data.Data object")
    if not isinstance(voronoi, bool):
        raise TypeError("voronoi must be a boolean")

    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        num_edges = 0
    elif graph.is_directed():
        num_edges = edge_index.size(1)
    else:
        num_edges = int((edge_index[0] < edge_index[1]).sum().item())

    adjacency = adjacency_matrix(
        graph,
        normalized = False,
        device = device,
        sparse = False,
    )

    fig, axes = plt.subplots(1, 2, figsize = figsize)
    plot_heatmap(
        adjacency,
        title = " ",
        ax = axes[0],
    )
    plot_nodes(
        graph,
        title = " ",
        ax = axes[1],
        voronoi = voronoi,
    )

    overall_title = title or f"n={graph.num_nodes}, edges={num_edges}"
    fig.suptitle(overall_title)
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    plt.show()


def plot_elbow(matrix: torch.Tensor, title: str = "Eigenvalue elbow plot",
               figsize: tuple[float, float] = (10, 5), symmetric: bool = True,
               max_ticks: int = 20) -> None:
    """
    Computes the eigenvalues of a square matrix and plots them in descending
    order as an elbow plot.

    Parameters:
    ----------
    matrix : torch.Tensor
        Square dense or sparse tensor whose eigenvalues are to be plotted.
        Sparse tensors are converted to dense before decomposition.
    title : str
        Title displayed above the plot.
    figsize : tuple[float, float]
        Figure width and height in inches.
    symmetric : bool
        Whether the matrix is symmetric (Hermitian for complex tensors). If
        True, ``torch.linalg.eigvalsh`` is used; otherwise,
        ``torch.linalg.eigvals`` is used and the real parts of the eigenvalues
        are plotted.
    max_ticks : int
        Maximum number of tick labels displayed on the x-axis.

    Returns:
    -------
    None
        The function displays a Matplotlib figure and does not return a value.

    Raises:
    ------
    ValueError
        If ``matrix`` is not a non-empty square matrix, ``symmetric`` is True
        but the matrix is not symmetric, or ``max_ticks`` is not positive.

    Notes:
    -----
    The matrix is detached from the computation graph before its spectral
    decomposition. For a non-symmetric matrix, eigenvalues may be complex; the
    plot follows the real component used to order them and omits the imaginary
    component.
    """
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1) or matrix.numel() == 0:
        raise ValueError("matrix must be a non-empty square tensor")
    if not isinstance(max_ticks, int) or isinstance(max_ticks, bool) or max_ticks < 1:
        raise ValueError("max_ticks must be a positive integer")

    dense_matrix = matrix.to_dense() if matrix.layout != torch.strided else matrix
    dense_matrix = dense_matrix.detach()
    if not (dense_matrix.is_floating_point() or dense_matrix.is_complex()):
        dense_matrix = dense_matrix.to(torch.get_default_dtype())

    if symmetric:
        if not torch.allclose(dense_matrix, dense_matrix.mH):
            raise ValueError("matrix must be symmetric when symmetric=True")
        eigenvalues = torch.linalg.eigvalsh(dense_matrix)
    else:
        eigenvalues = torch.linalg.eigvals(dense_matrix)

    eigenvalues = eigenvalues.real
    eigenvalues = eigenvalues[torch.argsort(eigenvalues, descending = True)]
    eigenvalue_values = eigenvalues.cpu().numpy()
    num_eigenvalues = eigenvalues.numel()
    eigenvalue_indices = range(1, num_eigenvalues + 1)
    tick_step = max(1, (num_eigenvalues + max_ticks - 1) // max_ticks)

    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(eigenvalue_indices, eigenvalue_values, marker = "o", markersize = 3)
    ax.set_title(title)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_xticks(range(1, num_eigenvalues + 1, tick_step))
    ax.tick_params(axis = "x", labelrotation = 45)
    ax.grid(alpha = 0.3)
    fig.tight_layout()
    plt.show()

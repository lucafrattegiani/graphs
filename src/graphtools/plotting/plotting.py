from collections.abc import Sequence
from copy import deepcopy
from math import ceil, sqrt

#Torch data and computations
import numpy as np
import torch
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from shapely.affinity import translate
from torch_geometric.data import Data

#Plotting
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..utils.matrices import adjacency_matrix

_VORONOI_FACE_COLOR = to_rgba("lightsteelblue", alpha = 0.8)
_VORONOI_EDGE_COLOR = to_rgba("tab:blue", alpha = 0.8)
_VORONOI_EDGE_ALPHA = 0.8
_VORONOI_EDGE_WIDTH = 1.0
_GRAPH_EDGE_COLOR = "black"
_GRAPH_NODE_COLOR = "red"
_GRAPH_NODE_FACE_ALPHA = 1
_GRAPH_EDGE_ZORDER = 1
_GRAPH_NODE_ZORDER = 2
_GRAPH_LEGEND_MARKER_SIZE = 8

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

    # Standard cells cover the whole plane, including infinite regions.
    # Use a separate patch: axis("off") hides the axes' own facecolor.
    # Axes coordinates keep the fill aligned when plot_positions changes limits.
    ax.add_artist(
        Rectangle(
            (0, 0), 1, 1,
            transform = ax.transAxes,
            facecolor = _VORONOI_FACE_COLOR,
            edgecolor = "none",
            zorder = 0,
        )
    )

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
                   voronoi: Voronoi | list[Polygon] | tuple[Voronoi | list[Polygon], ...] | None = None,
                   node_size: float | None = None,
                   names_labels: list[str] | None = None) -> None:
    """Plot two-dimensional positions with translucent fills and opaque borders.

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
        positions. A tuple may contain multiple independently positioned
        tessellations, as used when aligning two graphs.
    node_size : float | None, default = None
        Node marker area in points squared. If None, use
        ``max(70, 1800.0 / num_nodes)``.
    names_labels : list[str] | None, default = None
        Optional names shown in the legend, ordered according to the sorted
        unique values in ``labels``.

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

    unique_labels = np.unique(labels) if labels is not None else np.empty(0)
    if names_labels is not None:
        if not all(isinstance(name, str) for name in names_labels):
            raise TypeError("names_labels must contain only strings")
        if len(names_labels) != len(unique_labels):
            raise ValueError(
                "names_labels must contain one name for each unique label"
            )

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

    # Keep connections clearly visible even for dense graphs. Nodes are drawn
    # at a higher z-order below, so the lines never cover their markers.
    edge_alpha = min(0.8, max(0.15, 2000.0 / max(num_edges, 1)))
    edge_width = min(1.5, max(0.3, 600.0 / max(num_edges, 1)))
    if node_size is None:
        node_size = max(70, 1800.0 / num_nodes)

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

    tessellations = voronoi if isinstance(voronoi, tuple) else (voronoi,)
    for cells in tessellations:
        if isinstance(cells, Voronoi):
            plot_voronoi(cells, title = title, ax = ax)
        elif isinstance(cells, list):
            plot_epsilon_voronoi(cells, title = title, ax = ax)
            nonempty_cells = [cell for cell in cells if not cell.is_empty]
            if nonempty_cells:
                cell_bounds = np.asarray([cell.bounds for cell in nonempty_cells])
                plot_min = np.minimum(plot_min, cell_bounds[:, :2].min(axis = 0))
                plot_max = np.maximum(plot_max, cell_bounds[:, 2:].max(axis = 0))
        elif cells is not None:
            raise TypeError(
                "cells must be a scipy.spatial.Voronoi object, "
                "a list of shapely.geometry.Polygon objects, or None"
            )

    if num_edges > 0:
        ax.add_collection(
            LineCollection(
                segments,
                colors = _GRAPH_EDGE_COLOR,
                linewidths = edge_width,
                alpha = edge_alpha,
                zorder = _GRAPH_EDGE_ZORDER,
            )
        )

    if labels is None:
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            s = node_size,
            facecolors = to_rgba(_GRAPH_NODE_COLOR, alpha = _GRAPH_NODE_FACE_ALPHA),
            edgecolors = _GRAPH_NODE_COLOR,
            zorder = _GRAPH_NODE_ZORDER,
        )
    else:
        color_map = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
        legend_handles = []

        for color_index, class_id in enumerate(unique_labels):
            class_nodes = labels == class_id
            class_color = color_map(color_index % color_map.N)
            class_name = (
                names_labels[color_index]
                if names_labels is not None
                else str(class_id)
            )
            ax.scatter(
                nodes[class_nodes, 0],
                nodes[class_nodes, 1],
                s = node_size,
                facecolors = to_rgba(class_color, alpha = _GRAPH_NODE_FACE_ALPHA),
                edgecolors = class_color,
                zorder = _GRAPH_NODE_ZORDER,
            )
            legend_handles.append(
                Line2D(
                    [], [],
                    marker = "o",
                    linestyle = "none",
                    markersize = _GRAPH_LEGEND_MARKER_SIZE,
                    markerfacecolor = class_color,
                    markeredgecolor = class_color,
                    label = class_name,
                )
            )

        ax.legend(handles = legend_handles, title = "Labels")

    plot_title = title or f"n={num_nodes}, edges={num_edges}"
    ax.set_title(plot_title)
    ax.set_aspect("equal", adjustable = "box")
    ax.set_xlim(plot_min[0], plot_max[0])
    ax.set_ylim(plot_min[1], plot_max[1])
    ax.axis("off")
    if standalone:
        fig.tight_layout()
        plt.show()


def plot_node_alignment(graph_1: Data, graph_2: Data, coupling: torch.Tensor,
                        title: str = "Node alignment", figsize: tuple[float, float] = (12, 6),
                        ax: Axes | None = None, node_size: float | None = None,
                        linewidth: float = 2.0, voronoi: bool = False) -> None:
    """Plot two graphs' node positions and the coupling between their nodes.

    Parameters
    ----------
    graph_1, graph_2 : torch_geometric.data.Data
        Graphs with spatial coordinates in ``pos`` of shape (num_nodes, 2).
        Copies of the positions are vertically centered and separated
        horizontally, preserving each graph's geometry and scale.
    coupling : torch.Tensor
        Coupling matrix of shape (graph_1.num_nodes, graph_2.num_nodes).
        Entries must be finite and non-negative, with total mass one.
        Row sums must match graph_1.h and column sums graph_2.h, using
        uniform node probabilities when h is absent or None. Comparisons
        use rtol=1e-5 and atol=1e-8 to allow floating-point rounding errors.
        Positive entries are drawn as connections; zero entries are omitted.
    title : str, default = "Node alignment"
        Title displayed above the plot.
    figsize : tuple[float, float], default = (12, 6)
        Figure width and height in inches, used when ``ax`` is None.
    ax : matplotlib.axes.Axes | None, default = None
        Axes on which to draw. If None, create and display a new figure.
    node_size : float | None, default = None
        Node marker area in points squared, passed to ``plot_positions``.
    linewidth : float, default = 2.0
        Width in points of the strongest connection. Other widths are
        proportional to their coupling value relative to the maximum.

    voronoi : bool, default = False
        Draw graph.cells when present. Standard Voronoi objects and polygon
        lists are translated with the node positions on local copies.
        Missing cells are skipped; no tessellation is computed here.

    Returns
    -------
    None
        Draws the node alignment without visible axes. Graph positions are
        not modified. The figure is displayed only when ``ax`` is None.

    Raises
    ------
    ValueError
        If the coupling has invalid dimensions, entries, mass or marginals,
        or a graph histogram is not a probability vector of the right size.
    """
    # Validate the coupling before creating a figure or drawing on supplied axes.
    if not isinstance(coupling, torch.Tensor) or coupling.is_complex():
        raise TypeError("coupling must be a real torch.Tensor")
    expected_shape = (graph_1.num_nodes, graph_2.num_nodes)
    if coupling.shape != expected_shape:
        raise ValueError(f"coupling must have shape {expected_shape}, got {tuple(coupling.shape)}")
    weights = coupling.detach().to(device = "cpu", dtype = torch.float64)
    if weights.layout != torch.strided:
        weights = weights.to_dense()
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("coupling entries must be finite and non-negative")
    if not torch.isclose(weights.sum(), weights.new_tensor(1.0), rtol = 1e-5, atol = 1e-8):
        raise ValueError("coupling entries must sum to one")

    for index, (graph, marginal) in enumerate(
        ((graph_1, weights.sum(dim = 1)), (graph_2, weights.sum(dim = 0))), start = 1
    ):
        histogram = getattr(graph, "h", None)
        if histogram is None:
            histogram = weights.new_full((graph.num_nodes,), 1.0 / graph.num_nodes)
        else:
            histogram = torch.as_tensor(histogram, device = "cpu", dtype = weights.dtype)
            if histogram.shape != (graph.num_nodes,):
                raise ValueError(f"graph_{index}.h must have shape ({graph.num_nodes},)")
            if not torch.isfinite(histogram).all() or (histogram < 0).any():
                raise ValueError(f"graph_{index}.h must contain finite non-negative probabilities")
            if not torch.isclose(histogram.sum(), weights.new_tensor(1.0), rtol = 1e-5, atol = 1e-8):
                raise ValueError(f"graph_{index}.h probabilities must sum to one")
        if not torch.allclose(marginal, histogram, rtol = 1e-5, atol = 1e-8):
            axis = "Row" if index == 1 else "Column"
            raise ValueError(f"{axis} sums of coupling must match graph_{index}'s node histogram")

    if not isinstance(voronoi, bool):
        raise TypeError("voronoi must be a boolean")

    positions = [graph.pos.detach().cpu().clone() for graph in (graph_1, graph_2)]
    for pos in positions:
        pos[:, 0] -= pos[:, 0].min()
        pos[:, 1] -= (pos[:, 1].min() + pos[:, 1].max()) / 2
    span = max(float((pos.amax(dim = 0) - pos.amin(dim = 0)).max()) for pos in positions)
    positions[1][:, 0] += positions[0][:, 0].max() + 0.3 * max(span, 1.0)

    # Apply exactly the same translation to the stored cells as to the nodes.
    # Never recompute tessellations or modify graph.cells / graph.pos.
    shifted_cells = []
    if voronoi:
        for graph, pos in zip((graph_1, graph_2), positions):
            cells = getattr(graph, "cells", None)
            offset = (pos[0] - graph.pos[0].detach().cpu()).numpy()
            if isinstance(cells, Voronoi):
                cells = deepcopy(cells)
                cells.points[:] += offset
                cells.vertices[:] += offset
                cells.min_bound[:] += offset
                cells.max_bound[:] += offset
                shifted_cells.append(cells)
            elif isinstance(cells, list):
                shifted_cells.append([translate(cell, xoff=float(offset[0]),
                                                yoff=float(offset[1])) for cell in cells])
            elif cells is not None:
                raise TypeError("graph.cells must be a Voronoi object or a list of polygons")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize = figsize)
    else:
        fig = ax.figure

    # Plot both sets together so that plot_positions fits the full layout.
    labels = torch.cat([torch.full((len(pos),), index, dtype = torch.long)
                        for index, pos in enumerate(positions)])
    plot_positions(torch.cat(positions), ax = ax, title = title, labels = labels,
                   node_size = node_size, names_labels = ["Graph 1", "Graph 2"],
                   voronoi = tuple(shifted_cells) if voronoi else None)
    ax.get_legend().set_title("")

    source, target = torch.nonzero(weights > 0, as_tuple = True)
    if source.numel() > 0:
        segments = torch.stack([positions[0][source], positions[1][target]], dim = 1)
        widths = linewidth * weights[source, target] / weights.max()
        ax.add_collection(LineCollection(segments.numpy(), linewidths = widths.numpy(),
                                         colors = "gray", alpha = 0.6,
                                         zorder = _GRAPH_EDGE_ZORDER))
    ax.axis("off")

    if standalone:
        fig.tight_layout()
        plt.show()


def plot_nodes(graph: Data, title: str = "", ax: Axes | None = None,
               voronoi: bool = False, node_size: float | None = None,
               names_labels: list[str] | None = None) -> None:
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
    node_size : float | None, default = None
        Node marker area in points squared. If None, use
        ``max(70, 1800.0 / num_nodes)``.
    names_labels : list[str] | None, default = None
        Optional names shown in the legend, ordered according to the sorted
        unique values in ``graph.labels``.

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
        node_size = node_size,
        names_labels = names_labels,
    )


def plot_heatmap(matrix: torch.Tensor, title: str = "", figsize: tuple[float, float] = (7, 7),
                 ax: Axes | None = None, show_marginals: bool = False,
                 legend: bool = False) -> None:
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
    show_marginals : bool, default = False
        Plot column sums as vertical bars above the heatmap and row sums as
        horizontal bars to its left. Each bar is aligned with its matrix
        column or row and shows the actual sum, without normalization.

    legend : bool, default = False
        Display the colorbar mapping colors to matrix values.

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

    if show_marginals:
        # Keep the marginal bars aligned with the heatmap cells.
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size = "25%", pad = 0.15, sharex = ax)
        ax_left = divider.append_axes("left", size = "25%", pad = 0.65, sharey = ax)

        num_rows, num_columns = matrix_values.shape
        ax_top.bar(np.arange(num_columns), matrix_values.sum(axis = 0),
                   width = 1.0, color = "lightcoral", edgecolor = "black", linewidth = 0.5)
        ax_left.barh(np.arange(num_rows), matrix_values.sum(axis = 1),
                     height = 1.0, color = "lightsteelblue", edgecolor = "black", linewidth = 0.5)
        ax_left.invert_xaxis()
        ax.set_xlim(-0.5, num_columns - 0.5)
        ax.set_ylim(num_rows - 0.5, -0.5)

        ax.set_title("")
        ax_top.set_title(title)
        for marginal_ax in (ax_top, ax_left):
            marginal_ax.set_axis_off()

    if legend:
        if show_marginals:
            ax_colorbar = divider.append_axes("right", size = "5%", pad = 0.15)
            fig.colorbar(image, cax = ax_colorbar)
        else:
            fig.colorbar(image, ax = ax, fraction = 0.046, pad = 0.04)

    if standalone:
        fig.tight_layout()
        plt.show()


def plot_coupling_fgw(graph_1: Data, graph_2: Data, coupling: torch.Tensor,
                  title: str = "Coupling", figsize: tuple[float, float] = (15, 7),
                  node_size: float | None = None, linewidth: float = 2.0,
                  show_marginals: bool = False, voronoi: bool = False) -> None:
    """Plot a coupling heatmap and node alignment side by side.

    Parameters
    ----------
    graph_1, graph_2 : torch_geometric.data.Data
        Graphs with spatial coordinates in ``pos`` of shape (num_nodes, 2).
    coupling : torch.Tensor
        Coupling matrix of shape (graph_1.num_nodes, graph_2.num_nodes).
        Validated by ``plot_node_alignment``: finite non-negative entries,
        total mass one, and marginals matching each graph's ``h`` or uniform
        node probabilities when ``h`` is absent or None.
    title : str, default = "Coupling"
        Overall figure title.
    figsize : tuple[float, float], default = (15, 7)
        Figure width and height in inches.
    node_size : float | None, default = None
        Node marker area in points squared, passed to ``plot_node_alignment``.
    linewidth : float, default = 2.0
        Width in points of the strongest connection. Other widths are
        proportional to their coupling values.
    show_marginals : bool, default = False
        Whether to show column and row sums above and beside the heatmap.

    voronoi : bool, default = False
        Draw available graph.cells behind the aligned nodes. Passed to
        plot_node_alignment; cells are translated together with the nodes.

    Returns
    -------
    None
        Displays one figure with the heatmap on the left and node alignment
        on the right. The input graphs and coupling are not modified.
    """
    fig, axes = plt.subplots(1, 2, figsize = figsize)
    try:
        # Validate the coupling through the alignment plot before drawing its heatmap.
        plot_node_alignment(
            graph_1, graph_2, coupling,
            title = "Node alignment",
            ax = axes[1],
            node_size = node_size,
            linewidth = linewidth,
            voronoi = voronoi,
        )
        legend = axes[1].get_legend()
        if legend is not None:
            legend.remove()
        matrix = coupling.to_dense() if coupling.layout != torch.strided else coupling
        plot_heatmap(
            matrix,
            title = "Coupling matrix",
            ax = axes[0],
            show_marginals = show_marginals,
        )
    except Exception:
        plt.close(fig)
        raise

    fig.suptitle(title)
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    plt.show()


def plot_coupling_hungarian(
    batch_1: Sequence[Data], batch_2: Sequence[Data],
    assignments: torch.Tensor | Sequence[int], *,
    max_pairs: int | None = None,
    show_adjacency: bool = False,
    voronoi: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str = "Hungarian graph matching",
    node_size: float | None = None,
) -> None:
    """Plot matched graph pairs in a grid, using a previously computed assignment.

    Parameters:
    ----------
    batch_1, batch_2 : Sequence[torch_geometric.data.Data]
        Nonempty batches of equal length K. Node coordinates are read from
        pos and edges from edge_index. Graph sizes may differ. Node labels
        are displayed as in plot_nodes. Input graphs are not modified.
    assignments : torch.Tensor | Sequence[int]
        Integer permutation of 0, ..., K-1, such as info["assignment"] from
        hungarian_coupling. assignments[i] identifies the graph in batch_2
        matched to batch_1[i]. Matching is never computed inside this plot.
        GPU tensors are accepted and copied locally to CPU for indexing.
    max_pairs : int | None, default = None
        Display only the first max_pairs source graphs, in original batch
        order. None displays all pairs. Titles retain zero-based batch indices.
    show_adjacency : bool, default = False
        Draw each graph's adjacency matrix below its spatial plot. Matrices
        are computed on CPU from edge_index, not taken from graph.C.
        Colorbars are omitted, following plot_heatmap's default.
    voronoi : bool, default = False
        Draw the Voronoi cells stored in each graph.cells behind its nodes,
        using plot_nodes. Cells must already be available when enabled.
    figsize : tuple[float, float] | None, default = None
        Overall figure size in inches. If None, scale the figure to the grid.
        For n displayed pairs, columns = ceil(sqrt(n)) and
        rows = ceil(n / columns). Unused grid cells are hidden.
    title : str, default = "Hungarian graph matching"
        Overall figure title, displayed in bold. Each subplot title identifies
        the matched batch indices; individual graph titles are omitted.
    node_size : float | None, default = None
        Node marker area passed to plot_nodes; None uses its automatic size.

    Returns:
    -------
    None
        Displays a figure with two graphs per grid cell. Missing positions
        are handled by plot_nodes with a "No positions detected" placeholder.
        Font sizes scale with displayed pairs: 1x for 1-4, 1.4x for 5-9,
        1.8x for 10-16, and 2.2x above 16. Titles, ticks and legends are
        adjusted locally, without changing Matplotlib's global settings.
    """
    # Validate the supplied assignment before creating the figure.
    K = len(batch_1)
    if K == 0 or len(batch_2) != K:
        raise ValueError("Batches must be nonempty and have the same length.")
    if max_pairs is not None:
        if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs < 1:
            raise ValueError("max_pairs must be a positive integer.")
    indices = torch.as_tensor(assignments).detach().cpu()
    if indices.shape != (K,) or indices.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    ):
        raise ValueError("assignments must be a one-dimensional integer permutation.")
    indices = indices.to(torch.long)
    if not torch.equal(indices.sort().values, torch.arange(K)):
        raise ValueError("assignments must contain every target index exactly once.")

    # Each outer cell is a subfigure containing two spatial plots and,
    # optionally, two adjacency heatmaps directly underneath them.
    count = K if max_pairs is None else min(K, max_pairs)
    columns = ceil(sqrt(count))
    rows = ceil(count / columns)
    if figsize is None:
        figsize = (9 * columns, (8 if show_adjacency else 4.5) * rows)
    # Notebook displays shrink large figures; enlarge text in steps to
    # compensate. Use displayed pairs, so max_pairs also controls this scale.
    font_scale = 1.0 if count <= 4 else 1.4 if count <= 9 else 1.8 if count <= 16 else 2.2
    fig = plt.figure(figsize=figsize, layout="constrained")
    if title:
        heading = fig.suptitle(title, fontweight="bold")
        heading.set_fontsize(heading.get_fontsize() * font_scale)
    panels = fig.subfigures(rows, columns, squeeze=False)
    for i, panel in enumerate(panels.flat):
        if i >= count:
            panel.set_visible(False)
            continue
        j = int(indices[i])
        heading = panel.suptitle(f"Batch 1 [{i}] ↔ Batch 2 [{j}]")
        heading.set_fontsize(heading.get_fontsize() * font_scale)
        axes = panel.subplots(2 if show_adjacency else 1, 2, squeeze=False)
        for column, graph in enumerate((batch_1[i], batch_2[j])):
            # A blank space suppresses plot_nodes' automatic node/edge title.
            plot_nodes(graph, title=" ",
                       ax=axes[0, column], node_size=node_size, voronoi=voronoi)
            if show_adjacency:
                # adjacency_matrix requires at least two nodes; a singleton's
                # entry is determined directly by the presence of a self-loop.
                if graph.num_nodes == 1:
                    edges = getattr(graph, "edge_index", None)
                    adjacency = torch.tensor([[float(edges is not None and edges.numel() > 0)]])
                else:
                    adjacency = adjacency_matrix(graph, normalized=False,
                                                 device="cpu", sparse=False)
                plot_heatmap(adjacency, title="Adjacency", ax=axes[1, column])
        # Apply the same scale after drawing, including optional adjacency
        # ticks and the label legends created by plot_nodes.
        for ax in axes.flat:
            ax.title.set_fontsize(ax.title.get_fontsize() * font_scale)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(label.get_fontsize() * font_scale)
            legend = ax.get_legend()
            if legend is not None:
                for label in [legend.get_title(), *legend.get_texts()]:
                    label.set_fontsize(label.get_fontsize() * font_scale)
    plt.show()


def plot_graph(graph: Data, device: torch.device | str = "cpu", title: str = "",
               figsize: tuple[float, float] = (15, 7),
               voronoi: bool = False, node_size: float | None = None,
               names_labels: list[str] | None = None) -> None:
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
    node_size : float | None, default = None
        Node marker area in points squared. If None, use
        ``max(70, 1800.0 / num_nodes)``.
    names_labels : list[str] | None, default = None
        Optional names shown in the graph legend, ordered according to the
        sorted unique values in ``graph.labels``.

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
        node_size = node_size,
        names_labels = names_labels,
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

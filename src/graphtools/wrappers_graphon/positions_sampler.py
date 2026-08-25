#Mathematics and ML
import numpy as np
import torch
from torch_geometric.data import Data

#Plotting
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

#Clustering
from ..wrappers_metrics.spectral import spectral_clustering

#Polygonal structure
def polygonal(n: int, device: torch.device | str = "cpu", radius: float = 1.0, translation: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    """
    Generate n spatial coordinates placed on a regular polygon having n vertices.

    The polygon is centered at the translation vector and has the selected radius.

    Parameters
    ----------
    n : int
        Number of spatial coordinates to generate.
    device : torch.device | str, default = "cpu"
        Device on which to create the tensor.
    radius : float, default = 1.0
        Radius of the regular polygon.
    translation : tuple[float, float], default = (0.0, 0.0)
        Translation vector applied to the center of the polygon.

    Returns
    -------
    torch.Tensor
        Ordered spatial coordinates on the boundary of the translated polygon.
    """
    center = torch.as_tensor(translation, device = device)
    vertices = torch.arange(n, device = device)
    theta = 2.0 * torch.pi * vertices / n #Position on the polygon in radiants (as portions of the circumference of the circumscribed circle)
    offsets = radius * torch.stack([torch.cos(theta), torch.sin(theta)], dim = 1) #From polar coordinates to cartesian coordinates
    positions = center + offsets

    return positions

#Elliptic structure
def ellipse(n: int, device: torch.device | str = "cpu", amplitude: float = 1.0, elongation: float = 2.0, orientation: float = 1.0, translation: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    """
    Generate n spatial coordinates placed on the boundary of an ellipse.

    Parameters
    ----------
    n : int
        Number of spatial coordinates to generate.
    device : torch.device | str, default = "cpu"
        Device on which to create the tensor.
    amplitude : float, default = 1.0
        Base scale of the ellipse. It controls the minor semi-axis before rotation.
    elongation : float, default = 2.0
        Ratio between the major and minor semi-axes. Values larger than 1 stretch the ellipse.
    orientation : float, default = 1.0
        Rotation angle of the ellipse in radians.
    translation : tuple[float, float], default = (0.0, 0.0)
        Translation vector applied to the center of the ellipse.

    Returns
    -------
    torch.Tensor
        Ordered spatial coordinates on the boundary of the translated and rotated ellipse.
    """
    vertices = torch.arange(n, device = device)
    theta = 2.0 * torch.pi * vertices / n #Ordered positions on the circumference (radians)

    semi_major = amplitude * elongation #Horizontal semi-axis before rotation
    semi_minor = amplitude #Vertical semi-axis before rotation

    positions = torch.stack(
        [
            semi_major * torch.cos(theta),
            semi_minor * torch.sin(theta),
        ],
        dim = 1,
    )

    angle = torch.as_tensor(orientation, device = device)
    rotation = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)]),
        ]
    )

    center = torch.as_tensor(translation, device = device)
    positions = positions @ rotation.T #Rotate ellipse
    positions = positions + center #Translate ellipse

    return positions

#Random square structure
def square(n: int, device: torch.device | str = "cpu", lim_inf: float = -1.0, lim_sup: float = 1.0) -> torch.Tensor:
    """
    Generate n random spatial coordinates in the square [lim_inf, lim_sup] x [lim_inf, lim_sup].

    Parameters
    ----------
    n : int
        Number of spatial coordinates to generate.
    device : torch.device | str, default = "cpu"
        Device on which to create the tensor.
    lim_inf : float, default = -1.0
        Lower limit of the coordinate range.
    lim_sup : float, default = 1.0
        Upper limit of the coordinate range.

    Returns
    -------
    torch.Tensor
        Spatial coordinates in [lim_inf, lim_sup] x [lim_inf, lim_sup].
    """
    positions = torch.rand((n, 2), device = device)
    positions = (lim_sup - lim_inf) * positions + lim_inf

    return positions

#Random circle structure
def circle(n: int, device: torch.device | str = "cpu", translation: tuple[float, float] = (0.0, 0.0), radius: float = 1.0) -> torch.Tensor:
    """
    Generate n random spatial coordinates uniformly distributed inside a circle.

    The circle is centered at the translation vector.

    Parameters
    ----------
    n : int
        Number of spatial coordinates to generate.
    device : torch.device | str, default = "cpu"
        Device on which to create the tensor.
    translation : tuple[float, float], default = (0.0, 0.0)
        Translation vector applied to the center of the plotting window.
    radius : float, default = 1.0
        Radius of the circle.

    Returns
    -------
    torch.Tensor
        Spatial coordinates inside the translated circle.
    """
    center = torch.as_tensor(translation, device = device)
    theta = 2.0 * torch.pi * torch.rand(n, device = device) #Sample angles uniformly in [0, 2pi]
    radial_distance = radius * torch.sqrt(torch.rand(n, device = device)) #Sample distances from the center uniformly over [0, radius] (area-preserving transformation)
    
    #Transform polar coordinates to Cartesian coordinates
    offsets = torch.stack(
        [
            radial_distance * torch.cos(theta),
            radial_distance * torch.sin(theta),
        ],
        dim = 1,
    )

    #Translate to the selected center
    positions = center + offsets

    return positions

#Spectral bubble communities
def bubbles(n: int, device: torch.device | str = "cpu", radius: float = 0.2,
            K: int | None = None, graph: Data | None = None) -> torch.Tensor:
    """
    Generate node positions in bubbles determined by spectral clustering.

    Parameters
    ----------
    n : int
        Number of spatial coordinates to generate.
    device : torch.device | str, default = "cpu"
        Device on which to create the tensor.
    radius : float, default = 0.2
        Radius of each circular community.
    K : int | None, default = None
        Number of communities. If None, it is determined automatically from
        the maximum eigengap of the normalized adjacency matrix.
    graph : torch_geometric.data.Data | None, default = None
        Graph whose connectivity is used for spectral clustering.

    Returns
    -------
    torch.Tensor
        Spatial coordinates with shape ``[n, 2]``, aligned with the original
        node ordering.

    Raises
    ------
    TypeError
        If ``graph`` is not a ``torch_geometric.data.Data`` object.
    ValueError
        If the number of graph nodes differs from ``n`` or ``K`` is invalid.
    """
    if not isinstance(graph, Data):
        raise TypeError("graph must be a torch_geometric.data.Data object")
    if graph.num_nodes != n:
        raise ValueError("graph.num_nodes must be equal to n")

    assignments = spectral_clustering(
        graph,
        device = device,
        K = K,
    )
    communities = torch.unique(assignments, sorted = True)
    num_communities = communities.numel()

    if num_communities == 1:
        centers = torch.zeros((1, 2), device = device) #The unique center is at the origin
    else:
        centers = polygonal(num_communities, device) #Centers are vertices of a regular polygon

    positions = torch.empty((n, 2), device = device)
    for center_index, community in enumerate(communities):
        community_nodes = assignments == community
        community_positions = circle(
            int(community_nodes.sum().item()),
            device = device,
            translation = centers[center_index],
            radius = radius,
        )
        positions[community_nodes] = community_positions

    return positions

#Noisy polygonal structure
def ring(n: int, device: torch.device | str = "cpu", mode: str = "polygonal", amplitude: float = 1.0, elongation: float = 2.0, orientation: float = 1.0, translation: tuple[float, float] = (0.0, 0.0), radius: float = 1.0, noise_scale: float = 0.05, lim_inf: float = -1.0, lim_sup: float = 1.0) -> torch.Tensor:
    """
    Generate n spatial coordinates placed on a noisy regular polygon.

    The final noisy coordinates are rescaled to stay in [lim_inf, lim_sup].

    Parameters
    ----------
    n : int
        Number of spatial coordinates to generate.
    device : torch.device | str, default = "cpu"
        Device on which to create the tensor.
    mode : str, default = "polygonal"
        Method to generate the base positions. It can be "polygonal" or "ellipse".
    amplitude : float, default = 1.0
        Base scale of the ellipse. It controls the minor semi-axis before rotation.
    elongation : float, default = 2.0
        Ratio between the major and minor semi-axes. Values larger than 1 stretch the ellipse.
    orientation : float, default = 1.0
        Rotation angle of the ellipse in radians.
    translation : tuple[float, float], default = (0.0, 0.0)
        Translation vector applied to the center of the polygon or ellipse.
    radius : float, default = 1.0
        Radius of the regular polygon.
    noise_scale : float, default = 0.05
        Standard deviation of the noise added to each coordinate.

    Returns
    -------
    torch.Tensor
        Spatial coordinates.
    """
    #Select the method to generate the base positions
    if mode == "polygonal":
        positions = polygonal(n, device, radius = radius, translation = translation) #Position on the polygon
    elif mode == "ellipse":
        positions = ellipse(n, device, amplitude = amplitude, elongation = elongation, orientation = orientation, translation = translation) #Position on the ellipse

    #Perturb locations
    noise = noise_scale * torch.randn_like(positions) #Noise generation
    positions = positions + noise #Noise addition

    return positions

#Two-column structure
def two_columns(n: int, device: torch.device | str = "cpu", noise_scale: float = 0.05) -> torch.Tensor:
    """
    Generate n spatial coordinates placed on two noisy ordered vertical columns.

    The first n // 2 nodes are placed in the left column, ordered from top to bottom.
    The remaining nodes are placed in the right column, ordered from top to bottom.
    The final noisy coordinates are rescaled to stay in [lim_inf, lim_sup].

    Parameters
    ----------
    n : int
        Number of nodes.
    device : torch.device | str, default = "cpu"
        Device on which to create the tensor.
    noise_scale : float, default = 0.05
        Standard deviation of the noise added to each coordinate.
    Returns
    -------
    torch.Tensor
        Spatial coordinates.
    """
    size1 = n // 2 #Nodes in the first column
    size2 = n - size1 #Nodes in the second column

    y1 = torch.linspace(1.0, -1.0, steps = size1, device = device) #Equally spaced vertical coordinates
    y2 = torch.linspace(1.0, -1.0, steps = size2, device = device) #Equally spaced vertical coordinates

    x1 = torch.full((size1,), -1.0, device = device) #Common horizontal coordinate
    x2 = torch.full((size2,), 1.0, device = device) #Common horizontal coordinate

    first_column = torch.stack([x1, y1], dim = 1)
    second_column = torch.stack([x2, y2], dim = 1)

    positions = torch.cat([first_column, second_column], dim = 0)
    noise = noise_scale * torch.randn_like(positions) #Noise generation
    positions = positions + noise #Noise addition

    return positions

#Position sampling
def sample_positions(n: int, position_type: str, device: torch.device | str = "cpu", **kwargs) -> torch.Tensor:
    """
    Generate spatial positions from a specified position generator.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    position_type : str
        Position generator to use. It can be ``"ring"``, ``"bubbles"``,
        ``"square"``, ``"circle"``, or ``"two_columns"``.
    device : torch.device | str, default = "cpu"
        Device on which to generate the positions.
    **kwargs
        Additional arguments accepted by the selected position generator.

    Returns
    -------
    torch.Tensor
        Tensor containing ``n`` two-dimensional positions.

    Raises
    ------
    TypeError
        If ``n`` is not an integer or a keyword argument is not accepted by
        the selected position generator.
    ValueError
        If ``n`` is not positive or ``position_type`` is unsupported.

    Examples
    --------
    ``sample_positions(100, "ring", noise_scale = 0.1)``

    ``sample_positions(100, "bubbles", graph = graph, K = 4, radius = 0.15)``
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be positive")

    supported_types = ("ring", "bubbles", "square", "circle", "two_columns")
    if position_type not in supported_types:
        raise ValueError(
            f"Unsupported position_type {position_type!r}. "
            f"Choose one of: {', '.join(supported_types)}."
        )

    position_generators = {
        "ring": ring,
        "bubbles": bubbles,
        "square": square,
        "circle": circle,
        "two_columns": two_columns,
    }

    return position_generators[position_type](
        n = n,
        device = device,
        **kwargs,
    )

#Spatial graph plot
def plot_positions(graph: Data, title: str = "", ax: Axes | None = None,
                   labels: list[str] | tuple[str, ...] | None = None) -> None:
    """
    Plot a graph using the spatial coordinates stored in ``graph.pos``.

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
    labels : list[str] | tuple[str, ...] | None, default = None
        Optional class names indexed by the integer assignments stored in
        ``graph.labels``. If omitted, integer class identifiers are used in
        the legend.
    Returns
    -------
    None
        The function draws the graph and does not return a value. If
        ``graph.pos`` is missing, an empty plot titled
        ``"No positions detected"`` is drawn. The figure is displayed only
        when ``ax`` is None.

    Raises
    ------
    TypeError
        If ``graph`` is not a ``Data`` object, ``title`` is not a string, or
        ``graph.pos`` or ``graph.labels`` is not a tensor.
    ValueError
        If the graph has no nodes or the stored positions do not have shape
        ``[num_nodes, 2]``, or if the node labels are invalid.

    Examples
    --------
    ``graph.pos = sample_positions(graph.num_nodes, "ring")``

    ``plot_positions(graph)``
    """
    if not isinstance(graph, Data):
        raise TypeError("graph must be a torch_geometric.data.Data object")
    if not isinstance(title, str):
        raise TypeError("title must be a string")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize = (7, 7))
    else:
        fig = ax.figure

    positions = getattr(graph, "pos", None)
    if positions is None or (isinstance(positions, torch.Tensor) and positions.numel() == 0):
        ax.set_title("No positions detected")
        ax.axis("off")
        if standalone:
            fig.tight_layout()
            plt.show()
        return
    if not isinstance(positions, torch.Tensor):
        raise TypeError("graph.pos must be a torch.Tensor")

    num_nodes = graph.num_nodes
    if num_nodes is None or num_nodes < 1:
        raise ValueError("graph must contain at least one node")
    if positions.ndim != 2 or positions.shape != (num_nodes, 2):
        raise ValueError("graph.pos must have shape [num_nodes, 2]")

    nodes = positions.detach().cpu().numpy()

    node_labels = getattr(graph, "labels", None)
    if node_labels is not None:
        if not isinstance(node_labels, torch.Tensor):
            raise TypeError("graph.labels must be a torch.Tensor")
        if node_labels.ndim == 2 and node_labels.shape == (num_nodes, 1):
            node_labels = node_labels.squeeze(1)
        elif node_labels.ndim != 1 or node_labels.numel() != num_nodes:
            raise ValueError("graph.labels must have shape [num_nodes] or [num_nodes, 1]")
        if (
            node_labels.dtype == torch.bool
            or torch.is_floating_point(node_labels)
            or torch.is_complex(node_labels)
        ):
            raise TypeError("graph.labels must contain integers")
        node_labels = node_labels.detach().cpu().numpy()

    #Keep only one direction when an undirected edge is stored twice
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None and edge_index.numel() > 0:
        links = edge_index.detach().cpu().numpy()
        src_nodes = links[0]
        dst_nodes = links[1]

        if not graph.is_directed():
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

    #Adapt the visual parameters to the graph size
    edge_alpha = min(0.35, max(0.015, 800.0 / max(num_edges, 1)))
    edge_width = min(0.6, max(0.08, 200.0 / max(num_edges, 1)))
    node_size = max(35, max(3, 1000.0 / num_nodes))

    #Set plot limits from the actual node coordinates
    finite_nodes = nodes[np.isfinite(nodes).all(axis = 1)]
    if len(finite_nodes) == 0:
        raise ValueError("graph.pos must contain at least one finite coordinate pair")

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

    if node_labels is None:
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            s = node_size,
        )
    else:
        unique_labels = np.unique(node_labels)
        color_map = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")

        if labels is not None:
            if isinstance(labels, str):
                raise TypeError("labels must be a sequence of class names or None")
            if unique_labels.min() < 0 or unique_labels.max() >= len(labels):
                raise ValueError("labels must contain a name for every class identifier")

        for color_index, class_id in enumerate(unique_labels):
            class_nodes = node_labels == class_id
            class_name = str(class_id) if labels is None else str(labels[class_id])
            ax.scatter(
                nodes[class_nodes, 0],
                nodes[class_nodes, 1],
                s = node_size,
                color = color_map(color_index % color_map.N),
                label = class_name,
            )

        ax.legend(title = "Labels")
    graph_statistics = f"n={num_nodes}, edges={num_edges}"
    plot_title = title or graph_statistics
    ax.set_title(plot_title)
    ax.set_aspect("equal", adjustable = "box")
    ax.set_xlim(plot_min[0], plot_max[0])
    ax.set_ylim(plot_min[1], plot_max[1])
    ax.axis("off")
    if standalone:
        fig.tight_layout()
        plt.show()

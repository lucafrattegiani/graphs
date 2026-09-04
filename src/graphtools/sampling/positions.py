#Mathematics and ML
import torch
from torch_geometric.data import Data

#Clustering
from ..metrics.spectral import spectral_clustering

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

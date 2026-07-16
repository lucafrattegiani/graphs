#Mathematics and ML
import torch

#--------------------------------------------------
#GRAPHON FUNCTIONS
#--------------------------------------------------

#Ring structure
def ring_graphon(x: torch.Tensor, y: torch.Tensor, sigma: float = 0.08) -> torch.Tensor:
    """
    Ring-like graphon function. The function assigns high connection probability to pairs of nodes whose latent coordinates are close 
    with respect to the circular distance on [0, 1].

    Parameters
    ----------
    x : torch.Tensor
        Latent coordinates of the first set of nodes. It can be a scalar, vector, or any tensor broadcastable with `y`.
    y : torch.Tensor
        Latent coordinates of the second set of nodes. It must be broadcastable with `x`.
    sigma : float, default = 0.08
        Smaller values produce more local and usually sparser graphs. Larger values produce denser graphs.

    Returns
    -------
    torch.Tensor
        Connection probabilities in [0, 1].
    """
    distance = torch.abs(x - y) #Absolute distance 
    circular_distance = torch.minimum(distance, 1.0 - distance) #Circular distance
    probabilities =  torch.exp(-(circular_distance ** 2) / (2.0 * sigma ** 2)) #Connection probability

    return probabilities

#Communities (stochastic block model)
def sbm_graphon(x: torch.Tensor, y: torch.Tensor, block_probs: torch.Tensor = torch.tensor([[0.8, 0.1], [0.1, 0.8]])) -> torch.Tensor:
    """
    Multi-community stochastic block model graphon.

    Parameters
    ----------
    x, y:
        Latent coordinates in [0, 1].
    block_probs:
        Matrix of shape [K, K], where block_probs[a, b] is the connection probability between block a and block b.

    Returns
    -------
    torch.Tensor
        Connection probabilities in [0, 1].
    """
    K = block_probs.shape[0] #Extract number of blocks

    x_block = torch.clamp((x * K).long(), max = K - 1) #Assign every source node to a block according to its latent coordinate
    y_block = torch.clamp((y * K).long(), max = K - 1) #Assign every ending node to a block according to its latent coordinate

    #Return connection probabilities based on the block membership of source and ending node
    probabilities = block_probs[x_block, y_block]

    return probabilities

#Bipartite
def bipartite_graphon(x: torch.Tensor, y: torch.Tensor, p_between: float = 0.9, p_within: float = 0.05) -> torch.Tensor:
    """
    Bipartite-like graphon. The interval [0, 1] is split into two parts. Nodes in different parts connect with high probability, while nodes in the same part 
    connect with low probability.

    Parameters
    ----------
    x : torch.Tensor
        Latent coordinates of the first set of nodes.
    y : torch.Tensor
        Latent coordinates of the second set of nodes.
    p_between : float, default=0.8
        Connection probability between the two latent parts.
    p_within : float, default=0.05
        Connection probability within each latent part.

    Returns
    -------
    torch.Tensor
        Connection probabilities in [0, 1].
    """
    same_left = (x < 0.5) & (y < 0.5) #(source, end) couples belonging to the left side
    same_right = (x >= 0.5) & (y >= 0.5) #(source, end) couples belonging to the right side
    same_block = same_left | same_right #(source, end) couples belonging to the same side (boolean 'o' operation)

    #Connection probabilities according to side membership (low for same side, high for different side)
    probabilities = torch.where(
        same_block, #Check when nodes belong to the same side
        torch.as_tensor(p_within, device = x.device), #Low probability to same side
        torch.as_tensor(p_between, device = x.device), #High probability to different side
    )

    return probabilities

#--------------------------------------------------
#POSITION FUNCTIONS
#--------------------------------------------------

#Check limits
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

#Random bubble communities
def bubbles(n: int, device: torch.device | str = "cpu", radius: float = 0.2, K: int = 3) -> torch.Tensor:
    """
    Generate node positions in circular bubbles representing ordered communities.

    Parameters
    ----------
    n : int
        Number of spatial coordinates to generate.
    device : torch.device | str, default = "cpu"
        Device on which to create the tensor.
    K : int 
        Number of communities.
    radius : float, default = 0.2
        Radius of each circular community before the final rescaling.

    Returns
    -------
    torch.Tensor
        Spatial coordinates.
    """
    if K == 1:
        centers = torch.zeros((1, 2), device = device) #The unique center is at the origin
    else:
        centers = polygonal(K, device) #Multiple centers are vertices of K-polygon

    size = n // K #Equal size for each community
    remainder = n % K #Escluded nodes which number is in [0, ..., K - 1] (distributed among communities)
    community_sizes = torch.tensor(
        [size + int(community < remainder) for community in range(K)],
        device = device,
    )

    positions = []
    for community in range(K):
        current_size = size + int(community < remainder) #Progressively distribute rexcluded nodes among communities
        #Position of the k-th community
        current_positions = circle(
            current_size,
            device = device,
            translation = centers[community],
            radius = radius,
        )
        positions.append(current_positions)

    #Concatenate the whole set of positions
    positions = torch.cat(positions, dim = 0)

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
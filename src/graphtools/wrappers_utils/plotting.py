#Torch data and computations
import torch

#Plotting
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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
#Torch data and computations
from math import log

import torch
from ..validity import _floating_dtype

#--------------------------------------------------
#CENTRALIZATION
#--------------------------------------------------

def gini_index(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gini index for a vector of positive measuraments.

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor.

    Returns:
    -------
    torch.Tensor
        Gini index of the input vector.
    """
    n = x.numel() #Vector length
    if n == 0: #Nan values for empty tensors
        return torch.tensor(torch.nan, dtype = _floating_dtype(x), device = x.device)
    
    # Sort the measurements in ascending order
    sorted_x = torch.sort(x)[0]
    
    # Compute the Gini index
    cdf = torch.cumsum(sorted_x, dim = 0) / sorted_x.sum()
    gini_index = 1 - (1/n) * (cdf.sum() + cdf[:-1].sum())
    
    return gini_index

#--------------------------------------------------
#DISTANCE BETWEEN DISTRIBUTIONS
#--------------------------------------------------

def gaussian_kl_divergence(mu_i: torch.Tensor, sigma_i: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, diagonal: bool = False) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence between node-level Gaussian distributions and a global Gaussian distribution. Defined as:

    KL(N(mu_i, sigma_i) || N(mu, sigma))

    Parameters:
    ----------
    mu_i : torch.Tensor
        Node-level mean vectors with shape [n, d].
    sigma_i : torch.Tensor
        Node-level covariance parameters. Expected shape is [n, d] when diagonal is True and [n, d, d] otherwise.
    mu : torch.Tensor
        Global mean vector with shape [1, d].
    sigma : torch.Tensor
        Global covariance parameters. Expected shape is [d] when diagonal is True and [d, d] otherwise.
    diagonal : bool
        Whether to assume diagonal covariance matrices and use the optimized element-wise computation.

    Returns:
    -------
    torch.Tensor
        Kullback-Leibler divergence for each node with shape [n].
    """
    mean_difference = mu_i - mu

    if diagonal:
        kl_i = 0.5 * (
            sigma_i / sigma
            + mean_difference.square() / sigma
            - 1.0
            + torch.log(sigma / sigma_i)
        ).sum(dim = 1)

    else:
        num_nodes, num_features = mu_i.shape
        sigma_batch = sigma.unsqueeze(0).expand(num_nodes, -1, -1)

        trace_term = torch.diagonal(
            torch.linalg.solve(sigma_batch, sigma_i),
            dim1 = -2,
            dim2 = -1,
        ).sum(dim = 1)

        solved_mean_difference = torch.linalg.solve(
            sigma_batch,
            mean_difference.unsqueeze(2),
        ).squeeze(2)
        quadratic_term = (mean_difference * solved_mean_difference).sum(dim = 1)

        _, logdet_sigma = torch.linalg.slogdet(sigma)
        _, logdet_sigma_i = torch.linalg.slogdet(sigma_i)

        kl_i = 0.5 * (
            trace_term
            + quadratic_term
            - num_features
            + logdet_sigma
            - logdet_sigma_i
        )

    return kl_i.clamp_min(0.0)

def gaussian_hellinger_distance(mu_i: torch.Tensor, sigma_i: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, diagonal: bool = False) -> torch.Tensor:
    """
    Computes the squared Hellinger distance between node-level Gaussian distributions and a global Gaussian distribution. Defined as:

    H^2(N(mu_i, sigma_i), N(mu, sigma)) = 1 - BC(N(mu_i, sigma_i), N(mu, sigma))

    where BC is the Bhattacharyya coefficient.

    Parameters:
    ----------
    mu_i : torch.Tensor
        Node-level mean vectors with shape [n, d].
    sigma_i : torch.Tensor
        Node-level covariance parameters. Expected shape is [n, d] when diagonal is True and [n, d, d] otherwise.
    mu : torch.Tensor
        Global mean vector with shape [1, d].
    sigma : torch.Tensor
        Global covariance parameters. Expected shape is [d] when diagonal is True and [d, d] otherwise.
    diagonal : bool
        Whether to assume diagonal covariance matrices and use the optimized element-wise computation.

    Returns:
    -------
    torch.Tensor
        Squared Hellinger distance for each node with shape [n].
    """
    mean_difference = mu_i - mu

    if diagonal:
        average_sigma = 0.5 * (sigma_i + sigma)
        quadratic_term = (mean_difference.square() / average_sigma).sum(dim = 1)
        log_bc = (
            0.25 * torch.log(sigma_i).sum(dim = 1)
            + 0.25 * torch.log(sigma).sum()
            - 0.5 * torch.log(average_sigma).sum(dim = 1)
            - 0.125 * quadratic_term
        )

    else:
        num_nodes = mu_i.size(0)
        sigma_batch = sigma.unsqueeze(0).expand(num_nodes, -1, -1)
        average_sigma = 0.5 * (sigma_i + sigma_batch)

        solved_mean_difference = torch.linalg.solve(
            average_sigma,
            mean_difference.unsqueeze(2),
        ).squeeze(2)
        quadratic_term = (mean_difference * solved_mean_difference).sum(dim = 1)

        _, logdet_sigma = torch.linalg.slogdet(sigma)
        _, logdet_sigma_i = torch.linalg.slogdet(sigma_i)
        _, logdet_average_sigma = torch.linalg.slogdet(average_sigma)

        log_bc = (
            0.25 * logdet_sigma_i
            + 0.25 * logdet_sigma
            - 0.5 * logdet_average_sigma
            - 0.125 * quadratic_term
        )

    bhattacharyya_coefficient = torch.exp(log_bc.clamp(max = 0.0))
    return (1.0 - bhattacharyya_coefficient).clamp(min = 0.0, max = 1.0)


#--------------------------------------------------
#PAIRWISE NODE STATE COSTS
#--------------------------------------------------

def continuous_state_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes squared Euclidean distances between all pairs of rows.

    Parameters:
    ----------
    x, y : torch.Tensor
        Floating matrices with shapes [n, d] and [m, d], on the same device
        and with the same dtype.

    A leading batch dimension B is also supported: inputs [B, n, d] and
    [B, m, d] produce [B, n, m], comparing nodes within each batch pair.

    Returns:
    -------
    torch.Tensor
        Squared distance matrix with shape [n, m], on the input device.
    """
    return torch.cdist(x, y, p=2).square()


def soft_state_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Jensen-Shannon divergence in base 2.

    Parameters:
    ----------
    x, y : torch.Tensor
        Floating probability matrices with shapes [n, d] and [m, d], on the
        same device and with the same dtype. Rows must be finite, nonnegative
        and sum to one; columns must represent the same categories.

    A leading batch dimension B is also supported: inputs [B, n, d] and
    [B, m, d] produce [B, n, m], comparing nodes within each batch pair.

    Returns:
    -------
    torch.Tensor
        Divergence matrix with shape [n, m], on the input device. Values lie
        in [0, 1]; this is the squared Jensen-Shannon distance, without a root.
    """
    for probabilities in (x, y):
        if (probabilities.ndim not in (2, 3) or not torch.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not torch.allclose(probabilities.sum(dim=-1),
                                      probabilities.new_ones(probabilities.shape[:-1]))):
            raise ValueError("Soft features must be finite nonnegative probability rows summing to one.")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Soft features must have the same number of categories.")

    # Compute each row's entropy once. xlogy handles zero probabilities exactly.
    entropy_x = -torch.special.xlogy(x, x).sum(dim=-1)
    entropy_y = -torch.special.xlogy(y, y).sum(dim=-1)
    mixture = (x[..., :, None, :] + y[..., None, :, :]) * 0.5
    entropy_mixture = -torch.special.xlogy(mixture, mixture).sum(dim=-1)
    divergence = (entropy_mixture - 0.5 * (entropy_x[..., :, None] + entropy_y[..., None, :])) / log(2)
    return divergence.clamp_min(0) # Remove negative floating-point roundoff.


def hard_state_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise category agreement between one-hot rows.

    Parameters:
    ----------
    x, y : torch.Tensor
        One-hot matrices with shapes [n, d] and [m, d], on the same device
        and with the same dtype. Columns must represent the same categories.

    A leading batch dimension B is also supported: inputs [B, n, d] and
    [B, m, d] produce [B, n, m], comparing nodes within each batch pair.

    Returns:
    -------
    torch.Tensor
        Agreement matrix with shape [n, m], on the input device: 1 for equal
        categories, 0 otherwise. This is a similarity, not a dissimilarity;
        minimizing it as a transport cost favors different categories.
    """
    for assignments in (x, y):
        if (assignments.ndim not in (2, 3) or not ((assignments == 0) | (assignments == 1)).all()
                or not (assignments.sum(dim=-1) == 1).all()):
            raise ValueError("Hard features must contain one-hot rows.")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Hard features must have the same number of categories.")

    # One-hot dot products are 1 exactly when the selected categories agree.
    return x @ y.transpose(-2, -1)


def state_distance(x: torch.Tensor, y: torch.Tensor,
                   feature_type: str = "continuous") -> torch.Tensor:
    """
    Selects the pairwise state cost according to the feature type.

    Parameters:
    ----------
    x, y : torch.Tensor
        Matrices with shapes [n, d] and [m, d], on the same device and with
        the same dtype. See the selected function for input requirements.
    feature_type : str, default = "continuous"
        "continuous": squared Euclidean distance; "soft": Jensen-Shannon
        divergence in base 2; "hard": one-hot category agreement.

    A leading batch dimension B is also supported: inputs [B, n, d] and
    [B, m, d] produce [B, n, m], comparing nodes within each batch pair.

    Returns:
    -------
    torch.Tensor
        Pairwise cost matrix with shape [n, m], on the input device.
    """
    functions = {
        "continuous": continuous_state_distance,
        "soft": soft_state_distance,
        "hard": hard_state_distance,
    }
    if feature_type not in functions:
        raise ValueError('feature_type must be "continuous", "soft" or "hard".')
    return functions[feature_type](x, y)

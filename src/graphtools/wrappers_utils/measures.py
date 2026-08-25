#Torch data and computations
import torch

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
        return torch.tensor(torch.nan, device=x.device)
    
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
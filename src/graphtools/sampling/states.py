#Mathematics and ML
import torch
import torch.nn.functional as F

#Utilities
from ..utils.validity import _validate_size, _class_probabilities


def hard_state(n: int, K: int, device: torch.device | str = "cpu", one_hot: bool = True,
               p: list[float] | torch.Tensor = []) -> torch.Tensor:
    """
    Sample one discrete state for each node among K possible classes with specified probabilities.

    Parameters
    ----------
    n : int
        Number of nodes.
    K : int
        Number of possible classes.
    device : torch.device | str, default = "cpu"
        Device on which to generate the states.
    one_hot : bool, default = True
        If True, return a one-hot representation of the sampled classes.
    p : list[float] | torch.Tensor, default = []
        Probabilities of the ``K`` classes. If empty, all classes are sampled
        with equal probability.

    Returns
    -------
    torch.Tensor
        One-hot states with shape ``[n, K]`` and floating-point dtype when
        ``one_hot`` is True; otherwise, class indices with shape ``[n, 1]``.
    """
    _validate_size(n, "n")
    _validate_size(K, "K")
    if not isinstance(one_hot, bool):
        raise TypeError("one_hot must be a boolean")

    probabilities = _class_probabilities(K, p, device)
    x = torch.multinomial(probabilities, num_samples = n, replacement = True).unsqueeze(1)
    if one_hot:
        x = F.one_hot(x.squeeze(1), num_classes = K).to(torch.float)

    return x


def soft_state(n: int, K: int, device: torch.device | str = "cpu", p: list[float] | torch.Tensor = []) -> torch.Tensor:
    """
    Sample a categorical probability distribution for each node.

    Parameters
    ----------
    n : int
        Number of nodes.
    K : int
        Number of possible classes.
    device : torch.device | str, default = "cpu"
        Device on which to generate the states.
    p : list[float] | torch.Tensor, default = []
        Probabilities of the ``K`` classes. If empty, all classes have equal
        probability.

    Returns
    -------
    torch.Tensor
        Soft states with shape ``[n, K]`` whose rows sum to one.
    """
    _validate_size(n, "n")
    _validate_size(K, "K")

    probabilities = _class_probabilities(K, p, device)
    x = torch.rand(n, K, device = device) * probabilities
    x = x / x.sum(dim = 1, keepdim = True) #Normalize to get a probability distribution

    return x


def continuous_state(n: int, d: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Sample continuous node features from a standard normal distribution.

    Parameters
    ----------
    n : int
        Number of nodes.
    d : int
        Number of features for each node.
    device : torch.device | str, default = "cpu"
        Device on which to generate the states.

    Returns
    -------
    torch.Tensor
        Continuous states with shape ``[n, d]``.
    """
    _validate_size(n, "n")
    _validate_size(d, "d")

    x = torch.randn(n, d, device = device)

    return x


def sample_states(n: int, state_type: str, device: torch.device | str = "cpu",
                  **kwargs) -> torch.Tensor:
    """
    Sample one state for each node using the selected state generator.

    Parameters
    ----------
    n : int
        Number of nodes.
    state_type : str
        State generator to use. It can be ``"hard"``, ``"soft"``, or
        ``"continuous"``.
    device : torch.device | str, default = "cpu"
        Device on which to generate the states.
    **kwargs
        Additional arguments accepted by the selected state generator. For
        example, ``K``, ``one_hot``, and ``p`` for discrete states, or ``d``
        for continuous states.

    Returns
    -------
    torch.Tensor
        Tensor containing the state associated with each node.

    Raises
    ------
    ValueError
        If ``state_type`` is unsupported.
    TypeError
        If a required argument is missing or a keyword argument is not
        accepted by the selected state generator.

    Examples
    --------
    ``sample_states(100, "hard", K = 3, one_hot = True)``

    ``sample_states(100, "continuous", d = 8)``
    """
    state_generators = {
        "hard": hard_state,
        "soft": soft_state,
        "continuous": continuous_state,
    }
    if state_type not in state_generators:
        supported_types = ", ".join(state_generators)
        raise ValueError(
            f"Unsupported state_type {state_type!r}. "
            f"Choose one of: {supported_types}."
        )

    return state_generators[state_type](
        n = n,
        device = device,
        **kwargs,
    )

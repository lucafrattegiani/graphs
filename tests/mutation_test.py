"""Input preservation, independently of output values and placement contracts."""

import pytest
import torch

from graphtools import metrics, opt
from .utils import (
    DEVICES, METRIC_CASES, example_graph, snapshot, assert_graph_unchanged,
)


@pytest.mark.parametrize("function, kwargs, features", METRIC_CASES)
@pytest.mark.parametrize("input_device", DEVICES)
def test_metric_preserves_graph(function, kwargs, features, input_device):
    """Analysis preserves input tensors and metadata, including during local transfers."""
    graph = example_graph(features=features, device=input_device)
    graph.description = {"source": "test", "labels": [1, 2]}
    before = snapshot(graph)
    function(graph, device="cpu", **kwargs)
    assert_graph_unchanged(graph, before)


@pytest.mark.parametrize("provided_matrix", [None, "dense", "sparse"])
@pytest.mark.parametrize("input_device", DEVICES)
def test_fgw_preserves_both_graphs(provided_matrix, input_device):
    """FGW preserves both graphs, optional histograms and supplied structural matrices."""
    first = example_graph(torch.float64, device=input_device)
    second = example_graph()
    first.h = torch.tensor([.2, .3, .3, .2], dtype=torch.float64, device=input_device)
    if provided_matrix is not None:
        first.C = metrics.adjacency_matrix(first, sparse=provided_matrix == "sparse").double()
        second.C = metrics.adjacency_matrix(second, sparse=provided_matrix == "sparse")
    before_first, before_second = snapshot(first), snapshot(second)
    opt.fused_gromov_wasserstein_distance(first, second, device="cpu")
    assert_graph_unchanged(first, before_first)
    assert_graph_unchanged(second, before_second)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_gini_preserves_input(dtype):
    """Gini calculation leaves the original unsorted tensor unchanged."""
    values = torch.tensor([3, 1, 2], dtype=dtype)
    original, version = values.clone(), values._version
    metrics.gini_index(values)
    assert values._version == version
    torch.testing.assert_close(values, original, rtol=0, atol=0)


def test_feature_gradients_are_preserved():
    """Dirichlet backward computes gradients without changing feature values."""
    graph = example_graph(torch.float64, features="continuous")
    graph.x.requires_grad_()
    before = snapshot(graph)
    metrics.dirichlet_energy(graph).backward()
    assert torch.isfinite(graph.x.grad).all()
    assert_graph_unchanged(graph, before)

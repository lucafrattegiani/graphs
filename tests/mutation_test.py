# Testing library
import pytest

# Testing utilities
from .utils import (
    DEVICES, METRIC_CASES, example_graph, snapshot, assert_graph_unchanged,
)

# Data representation and computation
import torch

# Tested modules
from graphtools import metrics, opt


@pytest.mark.parametrize("input_device", DEVICES)
def test_spectral_isolate_remapping_preserves_graph(input_device):
    """Removing isolates internally preserves original node indices and attributes."""
    from torch_geometric.data import Data
    graph = Data(num_nodes=4, edge_index=torch.tensor([[1, 3], [3, 1]], device=input_device),
                 x=torch.arange(4, device=input_device).unsqueeze(1))
    before = snapshot(graph)
    metrics.spectral_clustering(graph, K=2, device="cpu")
    assert_graph_unchanged(graph, before)


@pytest.mark.parametrize("function, kwargs, features", METRIC_CASES)
@pytest.mark.parametrize("input_device", DEVICES)
def test_metric_mutation(function, kwargs, features, input_device):
    """Analysis preserves input tensors and metadata, including during local transfers."""
    graph = example_graph(features=features, device=input_device)
    graph.description = {"source": "test", "labels": [1, 2]}
    before = snapshot(graph)
    function(graph, device="cpu", **kwargs)
    assert_graph_unchanged(graph, before)


@pytest.mark.parametrize("provided_matrix", [None, "dense", "sparse"])
@pytest.mark.parametrize("input_device", DEVICES)
def test_fgw_mutation(provided_matrix, input_device):
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
def test_gini_mutation(dtype):
    """Gini calculation leaves the original unsorted tensor unchanged."""
    values = torch.tensor([3, 1, 2], dtype=dtype)
    original, version = values.clone(), values._version
    metrics.gini_index(values)
    assert values._version == version
    torch.testing.assert_close(values, original, rtol=0, atol=0)


def test_dirichlet_mutation():
    """Dirichlet backward computes gradients without changing feature values."""
    graph = example_graph(torch.float64, features="continuous")
    graph.x.requires_grad_()
    before = snapshot(graph)
    metrics.dirichlet_energy(graph).backward()
    assert torch.isfinite(graph.x.grad).all()
    assert_graph_unchanged(graph, before)


@pytest.mark.parametrize("input_device", DEVICES)
@pytest.mark.parametrize("alpha", [0., 0.5, 1.])
def test_hungarian_coupling_preserves_inputs(input_device, alpha):
    """Preparation and CPU assignment preserve graph attributes and tensor storage."""
    from .utils import coupling_batches
    first, second = coupling_batches(device=input_device)
    graphs = first + second
    for graph in graphs:
        graph.x.requires_grad_(True)
    before = [snapshot(graph) for graph in graphs]
    opt.hungarian_coupling(first, second, alpha=alpha, device="cpu", reg=1.)
    for graph, saved in zip(graphs, before):
        assert_graph_unchanged(graph, saved)

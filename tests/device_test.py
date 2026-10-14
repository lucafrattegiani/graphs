"""Device contracts: inferred placement, explicit transfers and CPU adapters."""

import pytest
import torch
from torch_geometric.data import Data

from graphtools import metrics, opt
from .utils import (
    CUDA, DEVICES, DEVICE_PAIRS, METRIC_CASES, FEATURE_CASES, TOPOLOGY_CASES,
    example_graph, output_tensors,
)


@pytest.mark.parametrize("function, kwargs, features", METRIC_CASES)
@pytest.mark.parametrize("input_device, output_device", DEVICE_PAIRS)
def test_metric_output_device(function, kwargs, features, input_device, output_device):
    """All computational graph metrics honor explicit and inferred output devices."""
    graph = example_graph(features=features, device=input_device)
    result = function(graph, device=output_device, **kwargs)
    expected = torch.device(input_device if output_device is None else output_device)
    if expected.type == "cuda":
        expected = torch.device("cuda", torch.cuda.current_device())
    assert all(tensor.device == expected for tensor in output_tensors(result))


@pytest.mark.parametrize("function, kwargs, features", FEATURE_CASES)
def test_feature_metrics_reject_mixed_devices(function, kwargs, features):
    """Relevant tensors on different devices require explicit transfer permission."""
    graph = example_graph(features=features)
    graph.x = graph.x.to("meta")
    with pytest.raises(ValueError, match="same device"):
        function(graph, **kwargs)


@CUDA
@pytest.mark.parametrize("function, kwargs, features", FEATURE_CASES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_device_resolves_mixed_inputs(function, kwargs, features, device):
    """An explicit device permits a metric to consume CPU edges and CUDA features."""
    graph = example_graph(features=features)
    graph.x = graph.x.cuda()
    result = function(graph, device=device, **kwargs)
    assert all(t.device.type == device for t in output_tensors(result))


@pytest.mark.parametrize("first_device, output_device", DEVICE_PAIRS)
@pytest.mark.parametrize("second_device", DEVICES)
def test_fgw_output_device(first_device, output_device, second_device):
    """FGW loss and every info tensor follow the request or the first graph."""
    first = example_graph(device=first_device)
    second = example_graph(device=second_device)
    # Supplied matrices and weights may require their own local CPU transfers.
    first.C = metrics.adjacency_matrix(first)
    second.h = torch.full((4,), .25, device=second_device)
    result = opt.fused_gromov_wasserstein_distance(first, second, device=output_device)
    expected = first_device if output_device is None else output_device
    assert all(t.device.type == expected for t in output_tensors(result))


@pytest.mark.parametrize("device", DEVICES)
def test_early_returns_honor_output_device(device):
    """Undefined centralities and empty-neighborhood matrices still honor placement."""
    graph = Data(num_nodes=1, edge_index=torch.empty((2, 0), dtype=torch.long))
    for function in (metrics.betweenness_centrality, metrics.harmonic_centrality,
                     metrics.pagerank_centrality):
        assert function(graph, device=device).device.type == device
    for function in (metrics.inverse_shortest_path_matrix, metrics.jaccard_similarity_matrix):
        for sparse in (False, True):
            assert function(graph, device=device, sparse=sparse).device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_gini_follows_tensor_device(device):
    """The tensor-only Gini utility follows its input without a device argument."""
    result = metrics.gini_index(torch.tensor([1., 2., 3.], device=device))
    assert result.device.type == device

"""Numerical contracts: dtype, known values, tolerances and solver invariants."""

import pytest
import torch
from torch_geometric.data import Data

from graphtools import metrics, opt
from .utils import CUDA, DEVICES, FEATURE_CASES, example_graph, output_tensors


@pytest.mark.parametrize("function, kwargs, features", FEATURE_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_feature_precision(function, kwargs, features, dtype):
    """Feature metrics preserve dtype and agree with a float64 reference calculation."""
    graph = example_graph(dtype, features=features)
    reference = example_graph(torch.float64, features=features)
    reference.x = graph.x.double()  # Compare the same input values at two precisions.
    result = function(graph, **kwargs)
    expected = function(reference, **kwargs)
    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    # Full covariance cases use eps=1e-3 in FEATURE_CASES to control conditioning.
    torch.testing.assert_close(result.double(), expected, rtol=2e-4, atol=2e-5)


@CUDA
@pytest.mark.parametrize("function, kwargs, features", FEATURE_CASES)
def test_feature_cpu_cuda_agreement(function, kwargs, features):
    """CPU and CUDA feature calculations agree within floating-point tolerances."""
    graph = example_graph(torch.float64, features=features)
    expected = function(graph, **kwargs)
    result = function(graph.clone().to("cuda"), **kwargs)
    torch.testing.assert_close(result.cpu(), expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("function", [
    metrics.hard_edge_homophily, metrics.adjusted_homophily_hard,
    metrics.jsd_informativeness,
])
def test_integer_features_use_float32(function):
    """Integer one-hot labels yield finite float32 metric values."""
    result = function(example_graph(torch.int64, features="hard"))
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("function", [
    metrics.edge_homophily, metrics.adjusted_homophily,
    metrics.neighborhood_informativeness,
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_undefined_results_preserve_dtype(function, dtype):
    """An edgeless singleton produces NaN in the feature dtype."""
    graph = Data(num_nodes=1, edge_index=torch.empty((2, 0), dtype=torch.long),
                 x=torch.ones((1, 2), dtype=dtype))
    result = function(graph)
    assert result.dtype == dtype
    assert torch.isnan(result)


@pytest.mark.parametrize("function, expected", [
    pytest.param(metrics.density, 2/3, id="density"),
    pytest.param(metrics.local_clustering_coeff, [1, 1, 1/3, 0], id="local-triangles"),
    pytest.param(metrics.global_clustering_coeff, 3/5, id="global-triangles"),
    pytest.param(metrics.betweenness_centrality, [0, 0, 2/3, 0], id="betweenness"),
    pytest.param(metrics.harmonic_centrality, [5/6, 5/6, 1, 2/3], id="harmonic"),
    pytest.param(metrics.hard_edge_homophily, 1/4, id="hard-homophily"),
])
def test_known_metric_values(function, expected):
    """Check independently known values on the triangle-with-tail graph."""
    result = function(example_graph(features="hard"))
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dirichlet_known_values(dtype):
    """Verify per-edge energies and their mean against explicit squared differences."""
    graph = example_graph(dtype)
    graph.x = torch.tensor([[0], [1], [3], [6]], dtype=dtype)
    expected = torch.tensor([.5, .5, 4.5, 4.5, 2, 2, 4.5, 4.5], dtype=dtype)
    torch.testing.assert_close(metrics.dirichlet_energy(graph, aggregate=False), expected)
    torch.testing.assert_close(metrics.dirichlet_energy(graph), torch.tensor(23/8, dtype=dtype))


@pytest.mark.parametrize("function, kwargs, expected", [
    pytest.param(metrics.adjacency_matrix, {"normalized": False},
                 [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], id="adjacency"),
    pytest.param(metrics.laplacian_matrix, {"normalized": False},
                 [[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]], id="laplacian"),
    pytest.param(metrics.random_walk_matrix, {"lazy": False},
                 [[0, 1, 0, 0], [.5, 0, .5, 0], [0, 1, 0, 0], [0, 0, 0, 0]], id="walk"),
    pytest.param(metrics.inverse_shortest_path_matrix, {},
                 [[1, .5, 1/3, 0], [.5, 1, .5, 0], [1/3, .5, 1, 0], [0, 0, 0, 1]], id="shortest-path"),
    pytest.param(metrics.jaccard_similarity_matrix, {},
                 [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]], id="jaccard"),
])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_known_matrices(function, kwargs, expected, sparse, device):
    """Check dense and sparse matrices on a path plus an isolated node."""
    graph = Data(num_nodes=4, edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
    result = function(graph, device=device, sparse=sparse, **kwargs)
    assert result.dtype == torch.float32
    assert result.is_sparse == sparse
    if sparse:
        assert result.is_coalesced()
    torch.testing.assert_close(result.to_dense().cpu(), torch.tensor(expected, dtype=torch.float32))


@pytest.mark.parametrize("matrix", ["adjacency", "laplacian", "random_walk"])
def test_spectrum_residual(matrix):
    """Validate A v = lambda v instead of comparing sign-ambiguous eigenvectors."""
    graph = example_graph()
    values, vectors = metrics.spectral_decomposition(graph, matrix=matrix)
    if matrix == "adjacency":
        operator = metrics.adjacency_matrix(graph, normalized=True, sparse=False)
    elif matrix == "laplacian":
        operator = metrics.laplacian_matrix(graph, normalized=True, sparse=False)
    else:
        operator = metrics.random_walk_matrix(graph, sparse=False)
    torch.testing.assert_close(operator.to(vectors.dtype) @ vectors,
                               vectors * values, rtol=1e-4, atol=1e-6)
    assert torch.all(values.real[:-1] >= values.real[1:])


def test_spectral_embedding_and_labels():
    """Check normalized embedding rows and cluster membership without fixing label IDs."""
    graph = example_graph()
    embedding = metrics.spectral_embedding(graph, K=2)
    assert embedding.shape == (4, 2)
    torch.testing.assert_close(embedding.norm(dim=1), torch.ones(4))
    labels = metrics.spectral_clustering(graph, K=2)
    assert labels.shape == (4,)
    assert labels.dtype == torch.long
    assert labels.unique().numel() == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gini_precision(dtype):
    """Check the known Gini value and the dtype of an empty-input NaN."""
    result = metrics.gini_index(torch.tensor([1, 2, 3], dtype=dtype))
    torch.testing.assert_close(result, torch.tensor(2/9, dtype=dtype))
    empty = metrics.gini_index(torch.empty(0, dtype=dtype))
    assert empty.dtype == dtype
    assert torch.isnan(empty)


@pytest.mark.parametrize("alpha", [0., .5, 1.])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fgw_analytic_cost_and_marginals(alpha, dtype):
    """Constant feature/structure costs give FGW=2-alpha for unequal graph sizes."""
    first = Data(num_nodes=2, x=torch.zeros((2, 2), dtype=dtype),
                 C=torch.zeros((2, 2), dtype=dtype), h=torch.tensor([.25, .75], dtype=dtype))
    second = Data(num_nodes=3, x=torch.ones((3, 2), dtype=dtype),
                  C=torch.ones((3, 3), dtype=dtype), h=torch.tensor([.2, .3, .5], dtype=dtype))
    value, info = opt.fused_gromov_wasserstein_distance(first, second, alpha=alpha)
    assert set(info) == {"T", "w_term", "gw_term"}
    assert all(t.dtype == dtype for t in output_tensors((value, info)))
    torch.testing.assert_close(value, torch.tensor(2-alpha, dtype=dtype))
    torch.testing.assert_close(info["w_term"], torch.tensor(2*(1-alpha), dtype=dtype))
    torch.testing.assert_close(info["gw_term"], torch.tensor(alpha, dtype=dtype))
    torch.testing.assert_close(value, info["w_term"] + info["gw_term"])
    assert torch.all(info["T"] >= -1e-7)
    torch.testing.assert_close(info["T"].sum(1), first.h)
    torch.testing.assert_close(info["T"].sum(0), second.h)


@pytest.mark.parametrize("attribute", ["x", "h", "C"])
def test_fgw_promotes_each_supplied_dtype(attribute):
    """Float64 data in either graph promotes the loss and all returned info tensors."""
    first, second = example_graph(), example_graph()
    second.C = metrics.adjacency_matrix(second).double() if attribute == "C" else None
    second.h = torch.full((4,), .25, dtype=torch.float64) if attribute == "h" else None
    if attribute == "x":
        second.x = second.x.double()
    result = opt.fused_gromov_wasserstein_distance(first, second)
    assert all(t.dtype == torch.float64 for t in output_tensors(result))


@pytest.mark.parametrize("structure", ["adjacency", "laplacian", "shortest_path", "jaccard"])
def test_pure_gw_ignores_features(structure):
    """Pure GW uses each structural builder without consuming unused meta features."""
    graph = example_graph()
    graph.x = torch.empty((4, 2), dtype=torch.float64, device="meta")
    value, info = opt.fused_gromov_wasserstein_distance(graph, graph, alpha=1, structure=structure)
    assert value.dtype == torch.float32
    assert torch.isfinite(value)
    torch.testing.assert_close(info["T"].sum(1), torch.full((4,), .25))
    torch.testing.assert_close(info["w_term"], torch.tensor(0.))


@pytest.mark.parametrize("device", DEVICES)
def test_pagerank_probability_mass(device):
    """PageRank remains a finite nonnegative probability distribution."""
    result = metrics.pagerank_centrality(example_graph(), device=device)
    assert torch.isfinite(result).all()
    assert torch.all(result >= 0)
    torch.testing.assert_close(result.sum(), result.new_tensor(1.))

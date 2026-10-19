"""Small shared inputs and explicit cases used by the three test categories."""

from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Data

from graphtools import metrics


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]
DEVICE_PAIRS = [
    pytest.param("cpu", None, id="cpu-inferred"),
    pytest.param("cpu", "cpu", id="cpu-explicit"),
    pytest.param("cpu", "cuda", marks=CUDA, id="cpu-to-cuda"),
    pytest.param("cuda", None, marks=CUDA, id="cuda-inferred"),
    pytest.param("cuda", "cpu", marks=CUDA, id="cuda-to-cpu"),
    pytest.param("cuda", "cuda", marks=CUDA, id="cuda-explicit"),
]

# Each case specifies the function, its options, and the appropriate node features.
# Explicit cases keep supported variants visible without inspecting function signatures.
FEATURE_CASES = [
    pytest.param(metrics.hard_edge_homophily, {}, "hard", id="hard-homophily"),
    pytest.param(metrics.soft_edge_homophily, {}, "soft", id="soft-jsd"),
    pytest.param(metrics.soft_edge_homophily, {"method": "euclidean"}, "soft", id="soft-dot"),
    pytest.param(metrics.continuous_edge_homophily, {}, "continuous", id="continuous-homophily"),
    pytest.param(metrics.adjusted_homophily_hard, {}, "hard", id="adjusted-hard"),
    pytest.param(metrics.adjusted_homophily_soft, {}, "soft", id="adjusted-soft"),
    pytest.param(metrics.edge_homophily, {"attributes_type": "hard"}, "hard", id="edge-hard"),
    pytest.param(metrics.edge_homophily, {"attributes_type": "soft"}, "soft", id="edge-soft"),
    pytest.param(metrics.edge_homophily, {"attributes_type": "continuous"}, "continuous", id="edge-continuous"),
    pytest.param(metrics.adjusted_homophily, {"attributes_type": "hard"}, "hard", id="adjusted-wrapper-hard"),
    pytest.param(metrics.adjusted_homophily, {"attributes_type": "soft"}, "soft", id="adjusted-wrapper-soft"),
    pytest.param(metrics.jsd_informativeness, {}, "hard", id="jsd-hard"),
    pytest.param(metrics.jsd_informativeness, {"level": "class"}, "hard", id="jsd-class"),
    pytest.param(metrics.jsd_informativeness, {"attributes_type": "soft", "aggregate": False}, "soft", id="jsd-soft-nodes"),
    pytest.param(metrics.continuous_neighbor_informativeness, {"eps": 1e-3}, "continuous", id="informativity-full"),
    pytest.param(metrics.continuous_neighbor_informativeness, {"diagonal": True, "eps": 1e-3}, "continuous", id="informativity-diagonal"),
    pytest.param(metrics.continuous_neighbor_informativeness, {"distance": "hellinger", "eps": 1e-3}, "continuous", id="informativity-hellinger"),
    pytest.param(metrics.neighborhood_informativeness, {"attributes_type": "hard"}, "hard", id="neighborhood-hard"),
    pytest.param(metrics.neighborhood_informativeness, {"attributes_type": "soft"}, "soft", id="neighborhood-soft"),
    pytest.param(metrics.neighborhood_informativeness, {"attributes_type": "continuous", "eps": 1e-3}, "continuous", id="neighborhood-continuous"),
    pytest.param(metrics.dirichlet_energy, {}, "continuous", id="dirichlet-mean"),
    pytest.param(metrics.dirichlet_energy, {"aggregate": False}, "continuous", id="dirichlet-edges"),
]

MATRIX_FUNCTIONS = [
    metrics.adjacency_matrix, metrics.laplacian_matrix, metrics.random_walk_matrix,
    metrics.inverse_shortest_path_matrix, metrics.jaccard_similarity_matrix,
]
TOPOLOGY_CASES = [
    pytest.param(metrics.density, {}, "soft", id="density"),
    pytest.param(metrics.local_clustering_coeff, {}, "soft", id="local-clustering"),
    pytest.param(metrics.global_clustering_coeff, {}, "soft", id="global-clustering"),
    pytest.param(metrics.betweenness_centrality, {}, "soft", id="betweenness"),
    pytest.param(metrics.harmonic_centrality, {}, "soft", id="harmonic"),
    pytest.param(metrics.pagerank_centrality, {}, "soft", id="pagerank"),
    pytest.param(metrics.centrality, {"method": "betweenness", "centralization": "freeman"}, "soft", id="centrality-freeman"),
    pytest.param(metrics.centrality, {"method": "harmonic", "centralization": "gini"}, "soft", id="centrality-gini"),
    pytest.param(metrics.centrality, {"method": "pagerank", "centralization": "entropy"}, "soft", id="centrality-entropy"),
    pytest.param(metrics.spectral_decomposition, {"matrix": "adjacency"}, "soft", id="spectrum-adjacency"),
    pytest.param(metrics.spectral_decomposition, {"matrix": "laplacian"}, "soft", id="spectrum-laplacian"),
    pytest.param(metrics.spectral_decomposition, {"matrix": "random_walk"}, "soft", id="spectrum-walk"),
    pytest.param(metrics.spectral_embedding, {"K": 2}, "soft", id="embedding"),
    pytest.param(metrics.spectral_clustering, {"K": 2}, "soft", id="spectral-clustering"),
    pytest.param(metrics.spectral_clustering, {}, "soft", id="spectral-clustering-auto"),
]
for function in MATRIX_FUNCTIONS:
    for sparse in (False, True):
        TOPOLOGY_CASES.append(pytest.param(
            function, {"sparse": sparse}, "soft",
            id=f"{function.__name__}-{'sparse' if sparse else 'dense'}",
        ))
METRIC_CASES = TOPOLOGY_CASES + FEATURE_CASES


def example_graph(dtype=torch.float32, features="soft", device="cpu"):
    """Build a triangle with a tail and valid hard, soft or continuous features."""
    values = {
        "hard": [[1, 0], [0, 1], [1, 0], [0, 1]],
        "soft": [[.8, .2], [.3, .7], [.6, .4], [.1, .9]],
        "continuous": [[.2, .1], [.9, .4], [.5, 1.3], [1.1, .8]],
    }
    return Data(
        num_nodes=4,
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2, 2, 3], [1, 0, 2, 0, 2, 1, 3, 2]], device=device),
        x=torch.tensor(values[features], dtype=dtype, device=device),
    )


def snapshot(graph):
    """Save tensor identity, values, versions and gradient flags, plus metadata."""
    return {
        name: (value, value.clone(), value._version, value.requires_grad)
        if isinstance(value, torch.Tensor) else deepcopy(value)
        for name, value in graph.to_dict().items()
    }


def assert_graph_unchanged(graph, before):
    """Require unchanged attributes, storage objects, values, device and dtype."""
    assert set(graph.keys()) == set(before)
    for name, saved in before.items():
        value = graph[name]
        if isinstance(saved, tuple) and saved and isinstance(saved[0], torch.Tensor):
            original, copy, version, requires_grad = saved
            assert value is original, name
            assert value._version == version, name
            assert value.requires_grad == requires_grad, name
            torch.testing.assert_close(value, copy, rtol=0, atol=0, equal_nan=True)
        else:
            assert value == saved, name


def output_tensors(result):
    """List tensors from a metric result or the FGW (loss, info) return value."""
    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, dict):
        return [tensor for value in result.values() for tensor in output_tensors(value)]
    if isinstance(result, tuple):
        return [tensor for value in result for tensor in output_tensors(value)]
    raise TypeError(f"Unexpected result type: {type(result).__name__}")


def coupling_batches(dtype=torch.float32, device="cpu"):
    """Two reversed graph batches with constant attributes and known FGW costs."""
    def graph(n, value):
        return Data(
            num_nodes=n,
            edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
            x=torch.full((n, 1), value, dtype=dtype, device=device),
            C=torch.full((n, n), value / 2, dtype=dtype, device=device),
            h=torch.arange(1, n + 1, dtype=dtype, device=device) / (n * (n + 1) / 2),
        )
    return ([graph(2, v) for v in (0., 2., 5.)],
            [graph(3, v) for v in (5., 0., 2.)])

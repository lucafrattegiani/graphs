# Testing library
import pytest

# Data representation and computation
import torch
from torch_geometric.data import Data

# Tested modules
from graphtools import metrics, opt

# Testing utilities
from .utils import CUDA, DEVICES, FEATURE_CASES, example_graph, output_tensors


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("case", ["empty", "singleton", "loops", "mixed", "random"])
def test_jaccard_torch_matches_neighborhood_counting(device, sparse, case):
    """Torch Jaccard agrees with CPU set counting, including empty and repeated edges."""
    from .utils import snapshot, assert_graph_unchanged

    if case == "empty":
        graph = Data(num_nodes=0, edge_index=None)
    elif case == "singleton":
        graph = Data(num_nodes=1, edge_index=torch.empty((2, 0), dtype=torch.long))
    elif case == "loops":
        graph = Data(num_nodes=3, edge_index=torch.tensor([[0, 1, 2], [0, 1, 2]]))
    elif case == "mixed":
        # One-way edges, duplicates, a self-loop and an isolated node.
        graph = Data(num_nodes=5, edge_index=torch.tensor([[0, 0, 1, 2, 2], [1, 1, 2, 3, 2]]))
    else:
        generator = torch.Generator().manual_seed(42)
        graph = Data(num_nodes=20, edge_index=torch.randint(20, (2, 70), generator=generator))
    # Independent reference from the definition, rather than a second Torch call.
    neighbors = [set() for _ in range(graph.num_nodes)]
    if graph.edge_index is not None:
        for source, target in graph.edge_index.t().tolist():
            if source != target:
                neighbors[source].add(target)
                neighbors[target].add(source)
    expected = torch.zeros((graph.num_nodes, graph.num_nodes))
    for i, first in enumerate(neighbors):
        for j, second in enumerate(neighbors):
            union = first | second
            expected[i, j] = len(first & second) / len(union) if union else 0.0
    graph = graph.to(device)
    before = snapshot(graph)
    result = metrics.jaccard_similarity_matrix(graph, device=device, sparse=sparse)
    assert result.device.type == device
    assert result.dtype == torch.float32
    assert result.is_sparse == sparse
    if sparse:
        assert result.is_coalesced()
        assert (result.values() > 0).all()
    torch.testing.assert_close(result.to_dense().cpu(), expected)
    assert_graph_unchanged(graph, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("radius_polygon", [0., 1., 2.5])
def test_bubbles_preserves_label_mapping(device, radius_polygon):
    """Interleaved and special labels map to the correct centers in node order."""
    from graphtools.sampling import positions

    assignments = torch.tensor([7, -1, 2, -2, 7, 2], device=device)
    result = positions.bubbles(6, communities=assignments, device=device,
                               radius_bubble=0, radius_polygon=radius_polygon)
    centers = positions.polygonal(4, device=device, radius=radius_polygon)
    torch.testing.assert_close(result, centers[torch.tensor([3, 1, 2, 0, 3, 2], device=device)])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("labels", [[0, 0, 0, 0], [0, 1, 2, 3]])
@pytest.mark.parametrize("radius_polygon", [0., 2.5])
def test_bubbles_offsets_stay_in_disk(device, labels, radius_polygon):
    """Bubble offsets respect their radius independently of the polygon radius."""
    from graphtools.sampling import positions

    assignments = torch.tensor(labels, device=device)
    result = positions.bubbles(4, communities=assignments, device=device,
                               radius_bubble=.2, radius_polygon=radius_polygon)
    centers = (torch.zeros((4, 2), device=device) if labels[0] == labels[-1]
               else positions.polygonal(4, device=device, radius=radius_polygon))
    assert result.shape == (4, 2)
    assert result.device.type == device
    assert torch.isfinite(result).all()
    assert ((result - centers).norm(dim=1) <= .2 + 1e-6).all()


def test_adjusted_hard_with_unused_classes():
    """Unused one-hot columns leave the degree-adjusted homophily unchanged."""
    graph = example_graph(torch.float64, features="hard")
    graph.x = torch.cat((graph.x, torch.zeros((4, 2), dtype=torch.float64)), dim=1)
    # One of four undirected edges joins equal classes; class degree totals are 5 and 3.
    result = metrics.adjusted_homophily_hard(graph, eps=0)
    torch.testing.assert_close(result, torch.tensor(-0.6, dtype=torch.float64))


def test_adjusted_soft_known_value():
    """A two-node graph gives observed overlap .38 and expected overlap .505."""
    graph = Data(num_nodes=2, edge_index=torch.tensor([[0, 1], [1, 0]]),
                 x=torch.tensor([[.8, .2], [.3, .7]], dtype=torch.float64))
    result = metrics.adjusted_homophily_soft(graph, eps=0)
    torch.testing.assert_close(result, torch.tensor(-25 / 99, dtype=torch.float64))


def test_adjusted_soft_rejects_negative_features():
    """Direct computation retains the rejection of negative soft probabilities."""
    graph = example_graph(features="soft")
    graph.x[0, 0] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        metrics.adjusted_homophily_soft(graph)


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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [0, 1, 4])
@pytest.mark.parametrize("K", [None, 1, 8])
def test_spectral_all_isolated(n, K, device):
    """Entirely isolated graphs get a distinct label per node, regardless of K."""
    graph = Data(num_nodes=n, edge_index=torch.empty((2, 0), dtype=torch.long))
    result = metrics.spectral_clustering(graph, K=K, device=device)
    torch.testing.assert_close(result, torch.arange(n, device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("K", [None, 1, 2])
def test_spectral_mixed_isolates(K, device):
    """Isolates retain -1 and do not affect clustering or automatic K selection."""
    graph = Data(num_nodes=5, edge_index=torch.tensor([[1, 3], [3, 1]]))
    result = metrics.spectral_clustering(graph, K=K, device=device)
    assert result.device.type == device
    assert (result[[0, 2, 4]] == -1).all()
    assert (result[[1, 3]] >= 0).all()
    assert result[[1, 3]].unique().numel() == (1 if K is None else K)


def test_spectral_k_excludes_isolates():
    """An explicit K cannot count isolated nodes as available cluster members."""
    graph = Data(num_nodes=5, edge_index=torch.tensor([[1, 3], [3, 1]]))
    with pytest.raises(ValueError, match="non-isolated"):
        metrics.spectral_clustering(graph, K=3)


@pytest.mark.parametrize("device", DEVICES)
def test_spectral_unusable_rows(monkeypatch, device):
    """Zero, NaN and infinite rows get -2; tiny usable rows still reach K-means."""
    from graphtools._metrics import spectral

    graph = Data(num_nodes=8, edge_index=torch.tensor(
        [[1, 2, 2, 3, 3, 4, 4, 5, 5, 6], [2, 1, 3, 2, 4, 3, 5, 4, 6, 5]]))

    def embedding(subgraph, K, device, normalized):
        assert subgraph.num_nodes == 6
        assert subgraph.edge_index.max() == 5
        assert not normalized
        return torch.tensor([[0., 0.], [float("nan"), 1.], [float("inf"), 0.],
                             [1e-12, 0.], [0., 1.], [1., 1.]], device=device)

    class CheckedKMeans:
        def __init__(self, **kwargs):
            assert kwargs["n_clusters"] == 2

        def __call__(self, coordinates):
            from types import SimpleNamespace
            assert coordinates.shape == (1, 3, 2)
            assert torch.isfinite(coordinates).all()
            torch.testing.assert_close(coordinates.norm(dim=-1), coordinates.new_ones((1, 3)))
            return SimpleNamespace(labels=torch.tensor([[0, 1, 1]], device=device))

    monkeypatch.setattr(spectral, "spectral_embedding", embedding)
    monkeypatch.setattr(spectral, "KMeans", CheckedKMeans)
    result = metrics.spectral_clustering(graph, K=2, device=device)
    torch.testing.assert_close(result, torch.tensor([-1, -2, -2, -2, 0, 1, 1, -1], device=device))


@pytest.mark.parametrize("remaining", [0, 1, 2])
def test_spectral_few_usable_nodes(monkeypatch, remaining):
    """Numerical exclusions can leave no nodes or require reducing K."""
    from graphtools._metrics import spectral

    def embedding(*args, **kwargs):
        rows = torch.zeros((4, 3))
        rows[:remaining, 0] = 1
        return rows

    monkeypatch.setattr(spectral, "spectral_embedding", embedding)
    labels = metrics.spectral_clustering(example_graph(), K=3)
    torch.testing.assert_close(labels[:remaining], torch.arange(remaining))
    assert (labels[remaining:] == -2).all()


@pytest.mark.parametrize("failure", ["solver", "nonfinite"])
def test_spectral_failed_spectrum(monkeypatch, failure):
    """Failed spectral analysis marks connected nodes -2 while keeping isolates -1."""
    from graphtools._metrics import spectral

    def decomposition(*args, **kwargs):
        if failure == "solver":
            raise torch.linalg.LinAlgError("eigensolver did not converge")
        return torch.tensor([float("nan"), 0.]), torch.eye(2)

    monkeypatch.setattr(spectral, "spectral_decomposition", decomposition)
    graph = Data(num_nodes=3, edge_index=torch.tensor([[0, 1], [1, 0]]))
    torch.testing.assert_close(metrics.spectral_clustering(graph), torch.tensor([-2, -2, -1]))


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


@pytest.mark.parametrize("alpha", [0., 0.5, 1.])
@pytest.mark.parametrize("pair_batch_size", [1, 4, 9])
def test_hungarian_coupling_known_costs(alpha, pair_batch_size):
    """All cross costs and the unique assignment match an analytic FGW example."""
    from .utils import coupling_batches
    first, second = coupling_batches(dtype=torch.float64)
    # Constant features/structures make the cost independent of the node plan.
    first[0].C = first[0].C.to_sparse()
    W, info = opt.hungarian_coupling(first, second, alpha=alpha,
                                  pair_batch_size=pair_batch_size, reg=1.)
    delta = torch.tensor([0., 2., 5.], dtype=torch.float64)[:, None] - torch.tensor([5., 0., 2.])
    expected = ((1 - alpha) + alpha / 4) * delta.square()
    torch.testing.assert_close(info["C"], expected)
    assert info["assignment"].tolist() == [1, 2, 0]
    torch.testing.assert_close(W.sum(0), W.new_full((3,), 1 / 3))
    torch.testing.assert_close(W.sum(1), W.new_full((3,), 1 / 3))
    assert torch.count_nonzero(W) == 3
    torch.testing.assert_close(info["cost"], W.new_zeros(()), atol=1e-12, rtol=0)
    assert W.dtype == torch.float64
    assert set(info) == {"C", "assignment", "cost"}


@pytest.mark.parametrize("alpha", [0., 1.])
def test_hungarian_coupling_ignores_unused_attributes(alpha):
    """Endpoint objectives require only their active attributes and dtype."""
    from .utils import coupling_batches
    first, second = coupling_batches()
    for graph in first + second:
        if alpha == 0:
            del graph.C
            del graph.h
        else:
            del graph.x
    W, info = opt.hungarian_coupling(first, second, alpha=alpha, reg=1.)
    for graph in first + second:
        unused = ("C", "h") if alpha == 0 else ("x",)
        for name in unused:
            graph[name] = torch.tensor([float("nan")], dtype=torch.float64)
    W_again, info_again = opt.hungarian_coupling(first, second, alpha=alpha, reg=1.)
    torch.testing.assert_close(W_again, W)
    torch.testing.assert_close(info_again["C"], info["C"])
    assert info_again["C"].dtype == torch.float32


@pytest.mark.parametrize("case", ["empty", "unequal", "node_counts", "missing_C", "feature_width", "masses"])
def test_hungarian_coupling_rejects_incompatible_batches(case):
    """Reject malformed batches before calling the numerical solver."""
    from .utils import coupling_batches
    first, second = coupling_batches()
    if case == "empty":
        first, second = [], []
    elif case == "unequal":
        second.pop()
    elif case == "node_counts":
        first[0].num_nodes = 4
    elif case == "missing_C":
        del first[0].C
    elif case == "feature_width":
        for graph in second:
            graph.x = graph.x.repeat(1, 2)
    else:
        first[0].h *= 2
    with pytest.raises(ValueError):
        opt.hungarian_coupling(first, second)


@pytest.mark.parametrize("alpha, expected", [(0., 2.), (0.5, 1.5), (1., 0.)])
def test_hungarian_coupling_node_masses(alpha, expected):
    """Nonuniform node masses affect FGW, while alpha zero uses uniform masses."""
    first = Data(num_nodes=2, C=torch.zeros(2, 2), x=torch.tensor([[0.], [2.]]),
                 h=torch.tensor([0.25, 0.75], dtype=torch.float64))
    second = Data(num_nodes=1, C=torch.zeros(1, 1), x=torch.zeros(1, 1))
    W, info = opt.hungarian_coupling([first], [second], alpha=alpha, reg=1.)
    torch.testing.assert_close(info["C"], info["C"].new_tensor([[expected]]))
    torch.testing.assert_close(W, W.new_ones((1, 1)))
    assert W.dtype == (torch.float32 if alpha == 0 else torch.float64)


@pytest.mark.parametrize("sparse", [False, True])
def test_prepare_graphs_support_and_plan_indices(sparse):
    """Filtering preserves original costs and lets a node plan regain its indices."""
    from graphtools.utils.preprocessing import prepare_graphs
    # The zero-mass middle node still contributes to the original path distance.
    C = torch.tensor([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    graph = Data(num_nodes=3, C=C.to_sparse() if sparse else C,
                 x=torch.tensor([[10.], [20.], [30.]]),
                 h=torch.tensor([0.25, 0., 0.75]))
    prepared = prepare_graphs(graph)
    assert prepared["active_nodes"].tolist() == [0, 2]
    assert prepared["removed_nodes"].tolist() == [1]
    torch.testing.assert_close(prepared["C"], torch.tensor([[0., 2.], [2., 0.]]))
    torch.testing.assert_close(prepared["x"], torch.tensor([[10.], [30.]]))
    reduced = torch.diag(prepared["h"])
    restored = reduced.new_zeros((prepared["num_nodes"], prepared["num_nodes"]))
    active = prepared["active_nodes"]
    restored[active[:, None], active[None, :]] = reduced
    torch.testing.assert_close(restored, torch.diag(graph.h))


@pytest.mark.parametrize("alpha", [0., 1.])
def test_prepare_graphs_endpoint_attributes(alpha):
    """Alpha zero ignores h/C; alpha one ignores x and preserves tiny positive masses."""
    from graphtools.utils.preprocessing import prepare_graphs
    graph = Data(num_nodes=3)
    if alpha == 0:
        graph.x = torch.ones(3, 1)
        graph.C = torch.tensor([float("nan")], dtype=torch.float64)
        graph.h = torch.zeros(3, dtype=torch.float64)
    else:
        graph.C = torch.eye(3, dtype=torch.float64)
        graph.h = torch.tensor([1e-12, 0., 1. - 1e-12], dtype=torch.float64)
        graph.x = torch.tensor([float("nan")])
    result = prepare_graphs(graph, alpha)
    if alpha == 0:
        assert result["removed_nodes"].numel() == 0
        torch.testing.assert_close(result["h"], torch.full((3,), 1 / 3))
        assert result["C"].dtype == torch.float32
    else:
        assert result["active_nodes"].tolist() == [0, 2]
        torch.testing.assert_close(result["h"], graph.h[[0, 2]], rtol=0, atol=0)
        assert result["x"] is None


@pytest.mark.parametrize("h", [[0., 0.], [-0.1, 1.1], [0.2, 0.2], [float("nan"), 1.]])
def test_prepare_graphs_invalid_masses(h):
    """An invalid histogram is rejected rather than silently normalized or filtered."""
    from graphtools.utils.preprocessing import prepare_graphs
    graph = Data(num_nodes=2, C=torch.eye(2), h=torch.tensor(h))
    with pytest.raises(ValueError, match="masses"):
        prepare_graphs(graph, alpha=1)


@pytest.mark.parametrize("alpha", [0., .5, 1.])
@pytest.mark.parametrize("pair_batch_size", [1, 3])
def test_hungarian_variable_sizes_match_independent_pairs(alpha, pair_batch_size):
    """Grouped costs match direct unpadded solves and yield a global optimum."""
    import itertools
    import ot
    from .utils import variable_coupling_batches
    first, second = variable_coupling_batches()
    settings = dict(reg=1., max_iter=12, tol=0., max_iter_inner=100, tol_inner=1e-9)
    W, info = opt.hungarian_coupling(first, second, alpha=alpha,
                                   pair_batch_size=pair_batch_size, **settings)
    expected = torch.empty(3, 3, dtype=torch.float64)
    # Independent extraction from original graphs checks filtering and the
    # scattering of all costs back to their original row/column positions.
    for i, g1 in enumerate(first):
        for j, g2 in enumerate(second):
            h1 = torch.full_like(g1.h, 1 / g1.num_nodes) if alpha == 0 else g1.h
            h2 = torch.full_like(g2.h, 1 / g2.num_nodes) if alpha == 0 else g2.h
            active1, active2 = h1 > 0, h2 > 0
            C1, C2 = g1.C[active1][:, active1], g2.C[active2][:, active2]
            if alpha == 0:
                C1, C2 = torch.zeros_like(C1), torch.zeros_like(C2)
            M = (torch.cdist(g1.x[active1], g2.x[active2]).square() if alpha < 1
                 else C1.new_zeros((C1.shape[0], C2.shape[0])))
            result = ot.batch.solve_gromov_batch(
                C1[None], C2[None], a=h1[active1][None], b=h2[active2][None],
                M=M[None], alpha=alpha, symmetric=True, grad="detach", **settings)
            expected[i, j] = result.value[0]
    torch.testing.assert_close(info["C"], expected, rtol=1e-6, atol=1e-8)
    minimum = min(sum(expected[i, p[i]] for i in range(3)) / 3
                  for p in itertools.permutations(range(3)))
    torch.testing.assert_close(info["cost"], minimum)
    torch.testing.assert_close(W.sum(0), W.new_full((3,), 1 / 3))
    torch.testing.assert_close(W.sum(1), W.new_full((3,), 1 / 3))


def test_prepare_batches_groups_by_support():
    """Grouping follows retained nodes, while preserving graph and node indices."""
    from graphtools.utils.preprocessing import prepare_batches
    from .utils import variable_coupling_batches
    first, _ = variable_coupling_batches()
    groups = prepare_batches(first, .5, device="cpu", dtype=torch.float64)
    assert set(groups) == {1, 2}
    assert groups[2]["indices"].tolist() == [0, 1]
    assert groups[1]["indices"].tolist() == [2]
    assert groups[2]["active_nodes"][0].tolist() == [0, 2]
    assert groups[2]["removed_nodes"][0].tolist() == [1, 3]
    assert groups[2]["num_nodes"] == [4, 2]
    assert groups[2]["C"].shape == (2, 2, 2)
    assert (groups[2]["h"] > 0).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n,K", [(12, 3), (10, 3), (4, 1), (4, 4)])
def test_bubbles_random_balanced_groups(device, n, K):
    """Random memberships occupy every bubble and differ in size by at most one."""
    from graphtools import graphon
    result = graphon.sample_positions(n, "bubbles", K=K, radius_bubble=0,
                                      radius_polygon=2.5, device=device)
    centers = (torch.zeros((1, 2), device=device) if K == 1
               else graphon.polygonal(K, device=device, radius=2.5))
    membership = torch.cdist(result, centers).argmin(1)
    counts = torch.bincount(membership, minlength=K)
    expected = torch.full((K,), n // K, device=device)
    expected[:n % K] += 1
    torch.testing.assert_close(counts, expected)
    torch.testing.assert_close(result, centers[membership])


def test_sample_graph_bubbles_uses_external_or_computed_communities(monkeypatch):
    """Only sample_graph computes communities, and explicit labels bypass clustering."""
    from graphtools import graphon
    labels = torch.tensor([1, 0, 1, 0])
    calls = []
    def cluster(graph, **kwargs):
        calls.append(graph)
        return labels
    monkeypatch.setattr(graphon, "spectral_clustering", cluster)
    automatic = graphon.sample_graph(4, positions="bubbles", position_kwargs={"radius_bubble": 0, "radius_polygon": 2.5})
    explicit = graphon.sample_graph(4, positions="bubbles",
                                    position_kwargs={"communities": labels, "radius_bubble": 0, "radius_polygon": 2.5})
    assert len(calls) == 1
    expected = graphon.polygonal(2, radius=2.5)[labels]
    torch.testing.assert_close(automatic.pos, expected)
    torch.testing.assert_close(explicit.pos, expected)
    torch.testing.assert_close(labels, torch.tensor([1, 0, 1, 0]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fgw_soft_cost_matrix_matches_jsd(monkeypatch, dtype):
    """All soft node-pair costs match JS divergence, including zero categories."""
    import ot
    from scipy.spatial.distance import jensenshannon
    p = torch.tensor([[1., 0., 0.], [.5, .5, 0.]], dtype=dtype)
    q = torch.tensor([[0., 1., 0.], [1., 0., 0.], [.2, .8, 0.]], dtype=dtype)
    expected = torch.tensor([[jensenshannon(a, b, base=2)**2
                              for b in q.double().numpy()] for a in p.double().numpy()], dtype=dtype)
    solve = ot.gromov.fused_gromov_wasserstein2
    def check_cost(*args, **kwargs):
        torch.testing.assert_close(kwargs["M"], expected, atol=1e-6, rtol=1e-5)
        return solve(*args, **kwargs)
    monkeypatch.setattr(ot.gromov, "fused_gromov_wasserstein2", check_cost)
    first = Data(num_nodes=2, x=p, C=torch.zeros(2, 2, dtype=dtype))
    second = Data(num_nodes=3, x=q, C=torch.zeros(3, 3, dtype=dtype))
    loss, info = opt.fused_gromov_wasserstein_distance(first, second, alpha=0, feature_type="soft")
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, (expected * info["T"]).sum())


@pytest.mark.parametrize("values", [[-.1, 1.1], [0., 0.], [.2, .2], [float("nan"), 1.]])
def test_fgw_soft_rejects_invalid_probabilities(values):
    """Soft costs reject invalid distributions instead of silently changing them."""
    graph = Data(num_nodes=1, x=torch.tensor([values]), C=torch.zeros(1, 1))
    with pytest.raises(ValueError, match="probability"):
        opt.fused_gromov_wasserstein_distance(graph, graph, feature_type="soft")


@pytest.mark.parametrize("feature_type, expected", [
    ("continuous", [[2., 0., 2.], [0., 2., 2.]]),
    ("soft", [[1., 0., 1.], [0., 1., 1.]]),
    ("hard", [[0., 1., 0.], [1., 0., 0.]]),
])
def test_state_distance_pairwise_costs(feature_type, expected):
    """Each state cost compares all row pairs with its documented convention."""
    from graphtools.utils import measures
    x = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    y = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    expected = torch.tensor(expected)
    direct = getattr(measures, f"{feature_type}_state_distance")
    torch.testing.assert_close(direct(x, y), expected)
    torch.testing.assert_close(measures.state_distance(x, y, feature_type), expected)


@pytest.mark.parametrize("feature_type", ["continuous", "soft", "hard"])
@pytest.mark.parametrize("alpha", [0., .5, 1.])
def test_hungarian_state_costs(monkeypatch, feature_type, alpha):
    """Chunks pass the selected node costs to POT and preserve graph-pair order."""
    import ot
    from graphtools.utils.measures import state_distance
    first = [Data(x=torch.tensor([[1., 0.], [0., 1.]]), C=torch.zeros(2, 2)),
             Data(x=torch.tensor([[0., 1.], [0., 1.]]), C=torch.zeros(2, 2))]
    second = [Data(x=torch.tensor([[0., 1.], [1., 0.], [1., 0.]]), C=torch.zeros(3, 3)),
              Data(x=torch.tensor([[1., 0.], [1., 0.], [1., 0.]]), C=torch.zeros(3, 3))]
    if feature_type == "soft":
        for graph in first + second:
            graph.x = .8 * graph.x + .1
    expected = torch.stack([state_distance(a.x, b.x, feature_type)
                            for a in first for b in second])
    if alpha == 1:
        expected.zero_()
        for graph in first + second:
            graph.num_nodes = graph.x.shape[0]
            graph.x = None # Structural-only solves must not access features.
    solve = ot.batch.solve_gromov_batch
    seen = 0
    def check_cost(*args, **kwargs):
        nonlocal seen
        size = kwargs["M"].shape[0]
        torch.testing.assert_close(kwargs["M"], expected[seen:seen + size])
        seen += size
        return solve(*args, **kwargs)
    monkeypatch.setattr(ot.batch, "solve_gromov_batch", check_cost)
    W, info = opt.hungarian_coupling(first, second, alpha=alpha,
                                    feature_type=feature_type, pair_batch_size=3,
                                    reg=.1, verbose=False)
    assert seen == 4
    assert torch.isfinite(info["C"]).all()
    torch.testing.assert_close(W.sum(0), torch.full((2,), .5))
    torch.testing.assert_close(W.sum(1), torch.full((2,), .5))

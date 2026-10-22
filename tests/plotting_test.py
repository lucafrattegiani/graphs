"""Probability labels and matching node/cell colors, without opening windows."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from torch_geometric.data import Data

from graphtools.plotting import set_colors, color_labels, plot_nodes, plot_node_alignment


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_probability_colors():
    """Pure categories remain stable across K; soft colors are RGB mixtures."""
    colors = set_colors(3)
    torch.testing.assert_close(colors[:2], set_colors(2))
    labels = torch.tensor([[1., 0., 0.], [.25, .75, 0.]])
    before = labels.clone()
    result = color_labels(labels, colors)
    torch.testing.assert_close(result[0], colors[0])
    torch.testing.assert_close(result[1], .25 * colors[0] + .75 * colors[1])
    torch.testing.assert_close(labels, before)
    with pytest.raises(ValueError, match="only"):
        set_colors(21)


@pytest.mark.parametrize("labels", [torch.tensor([0, 1]), torch.tensor([[.2, .2]]),
                                   torch.tensor([[-.1, 1.1]]),
                                   torch.tensor([[float("nan"), 1.]])])
def test_invalid_color_labels(labels):
    """Integer vectors and malformed probability rows are rejected."""
    with pytest.raises(ValueError):
        color_labels(labels, set_colors(2))


def test_epsilon_cell_colors_and_legend():
    """Cells retain node indices across empty polygons and share node RGB."""
    labels = torch.tensor([[1., 0.], [.5, .5], [0., 1.]])
    graph = Data(edge_index=torch.empty((2, 0), dtype=torch.long),
                 pos=torch.tensor([[0., 0.], [1., 0.], [2., 0.]]), labels=labels,
                 cells=[box(-.3, -.3, .3, .3), Polygon(), box(1.7, -.3, 2.3, .3)])
    fig, ax = plt.subplots()
    plot_nodes(graph, ax=ax, voronoi=True, labels_names=["A", "B"])
    expected = color_labels(labels, set_colors(2)).numpy()
    np.testing.assert_allclose(ax.collections[-1].get_facecolors()[:, :3], expected)
    for patch, index in zip(ax.patches, [0, 2]):
        np.testing.assert_allclose(patch.get_facecolor()[:3], expected[index])
        assert patch.get_facecolor()[3] < ax.collections[-1].get_facecolors()[index, 3]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B"]


def test_standard_voronoi_colors_include_infinite_cells():
    """Both finite and infinite cells are colored in generating-node order."""
    points = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.], [1., 1.]])
    cells = Voronoi(points)
    original_vertices = cells.vertices.copy()
    labels = torch.eye(5)
    graph = Data(edge_index=torch.empty((2, 0), dtype=torch.long),
                 pos=torch.tensor(points), cells=cells, labels=labels)
    fig, ax = plt.subplots()
    plot_nodes(graph, ax=ax, voronoi=True)
    assert len(ax.patches) == 5
    colors = set_colors(5).numpy()
    for i, patch in enumerate(ax.patches):
        np.testing.assert_allclose(patch.get_facecolor()[:3], colors[i])
        # Every clipped polygon belongs to its generating site's half-planes.
        polygon = patch.get_xy()
        distances = ((polygon[:, None, :] - points[None, :, :])**2).sum(-1)
        assert np.all(distances[:, i:i+1] <= distances + 1e-8)
    np.testing.assert_array_equal(cells.vertices, original_vertices)


def test_alignment_missing_cells_preserves_colors():
    """A graph without cells must not shift the second graph's cell colors."""
    first = Data(pos=torch.tensor([[0., 0.], [1., 0.]]), num_nodes=2)
    second = Data(pos=torch.tensor([[0., 0.], [1., 0.]]), num_nodes=2,
                  cells=[box(-.2, -.2, .2, .2), box(.8, -.2, 1.2, .2)])
    fig, ax = plt.subplots()
    plot_node_alignment(first, second, torch.eye(2) / 2, ax=ax, voronoi=True)
    for patch in ax.patches:
        np.testing.assert_allclose(patch.get_facecolor()[:3], set_colors(2)[1].numpy())

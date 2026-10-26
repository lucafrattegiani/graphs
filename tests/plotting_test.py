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
    """Pure categories stay stable; entropy shades only the dominant category."""
    colors = set_colors(3)
    torch.testing.assert_close(colors[:2], set_colors(2))
    labels = torch.tensor([[1., 0., 0.], [.25, .75, 0.]])
    before = labels.clone()
    result = color_labels(labels, colors)
    torch.testing.assert_close(result[0], colors[0])
    uncertainty = -(labels[1] * labels[1].clamp_min(1e-30).log()).sum() / np.log(3)
    torch.testing.assert_close(result[1], (1 - uncertainty) * colors[1] + uncertainty * .85)
    # Uniform assignments are neutral, while a single category stays pure.
    torch.testing.assert_close(color_labels(torch.full((1, 3), 1 / 3), colors),
                               torch.full((1, 3), .85))
    torch.testing.assert_close(color_labels(torch.ones(2, 1), colors[:1]),
                               colors[:1].expand(2, -1))
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
        assert patch.get_facecolor()[3] == ax.collections[-1].get_facecolors()[index, 3] == 1
    np.testing.assert_allclose(ax.collections[-1].get_edgecolors(), [[0., 0., 0., 1.]])
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
                  labels=torch.tensor([[0., 1.], [0., 1.]]),
                  cells=[box(-.2, -.2, .2, .2), box(.8, -.2, 1.2, .2)])
    fig, ax = plt.subplots()
    plot_node_alignment(first, second, torch.eye(2) / 2, ax=ax, voronoi=True)
    for patch in ax.patches:
        np.testing.assert_allclose(patch.get_facecolor()[:3], set_colors(2)[1].numpy())


def test_fgw_alignment_uses_graph_labels(monkeypatch):
    """The FGW wrapper propagates category names and colors to nodes and cells."""
    from matplotlib.collections import PathCollection
    from graphtools.plotting import plot_coupling_fgw
    monkeypatch.setattr(plt, "show", lambda: None)
    graphs = [Data(pos=torch.tensor([[0., 0.], [1., 0.]]), num_nodes=2,
                   labels=labels, cells=[box(-.2, -.2, .2, .2), box(.8, -.2, 1.2, .2)])
              for labels in (torch.tensor([[1., 0.], [.25, .75]]),
                             torch.tensor([[0., 1.], [.6, .4]]))]
    positions = [g.pos.clone() for g in graphs]
    labels = [g.labels.clone() for g in graphs]
    plot_coupling_fgw(*graphs, torch.eye(2) / 2, voronoi=True,
                      labels_names=["A", "B"], palette="tab10")
    ax = plt.gcf().axes[1]
    nodes = [c for c in ax.collections if isinstance(c, PathCollection)]
    for index, (graph, collection) in enumerate(zip(graphs, nodes)):
        expected = color_labels(graph.labels, set_colors(2, "tab10")).numpy()
        np.testing.assert_allclose(collection.get_facecolors()[:, :3], expected)
        for patch, color in zip(ax.patches[2*index:2*index+2], expected):
            np.testing.assert_allclose(patch.get_facecolor()[:3], color)
        torch.testing.assert_close(graph.pos, positions[index])
        torch.testing.assert_close(graph.labels, labels[index])
    assert len(nodes) == 2
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["A", "B"]


def test_alignment_without_labels_has_no_category_legend():
    """Unlabelled graphs no longer acquire artificial graph-identity labels."""
    graph = Data(pos=torch.tensor([[0., 0.], [1., 0.]]), num_nodes=2)
    fig, ax = plt.subplots()
    plot_node_alignment(graph, graph, torch.eye(2) / 2, ax=ax)
    assert ax.get_legend() is None


@pytest.mark.parametrize("labelled", [True, False])
def test_fgw_marginal_colors_follow_nodes(monkeypatch, labelled):
    """Rectangular coupling bars preserve node order, entropy shading and masses."""
    from matplotlib.collections import PathCollection
    from graphtools.plotting import plot_coupling_fgw
    monkeypatch.setattr(plt, "show", lambda: None)
    first = Data(pos=torch.tensor([[0., 0.], [1., 0.]]), num_nodes=2)
    second = Data(pos=torch.tensor([[0., 0.], [1., 0.], [2., 0.]]), num_nodes=3)
    if labelled:
        first.labels = torch.tensor([[1., 0.], [.3, .7]])
        second.labels = torch.tensor([[0., 1.], [.8, .2], [1., 0.]])
    coupling = torch.full((2, 3), 1 / 6)
    plot_coupling_fgw(first, second, coupling, show_marginals=True, palette="tab10")
    _, alignment, top, left = plt.gcf().axes
    nodes = [c for c in alignment.collections if isinstance(c, PathCollection)]
    for bars, collection, masses in ((left.patches, nodes[0], coupling.sum(1)),
                                      (top.patches, nodes[1], coupling.sum(0))):
        colors = collection.get_facecolors()
        if len(colors) == 1:
            colors = np.repeat(colors, len(bars), axis=0)
        np.testing.assert_allclose([bar.get_facecolor() for bar in bars], colors)
        sizes = [bar.get_width() if bar.axes is left else bar.get_height() for bar in bars]
        np.testing.assert_allclose(sizes, masses.numpy())


def test_heatmap_probability_scale():
    """Selected probability uses a fixed white-to-category scale, not argmax."""
    labels = torch.tensor([[1., 0.], [.25, .75], [0., 1.]])
    colors = set_colors(2)
    expected = .25 + .75 * colors[1]
    result = color_labels(labels, colors, mode="heatmap", label_index=1)
    torch.testing.assert_close(result[0], torch.ones(3))
    torch.testing.assert_close(result[1], expected)
    torch.testing.assert_close(result[2], colors[1])
    with pytest.raises(ValueError, match="label_index"):
        color_labels(labels, colors, mode="heatmap", label_index=2)


def test_fgw_heatmap_selection_updates_all_artists(monkeypatch):
    """One selection updates both graphs, cells and bars, leaving the matrix intact."""
    from matplotlib.collections import PathCollection
    from graphtools.plotting import plot_coupling_fgw
    monkeypatch.setattr(plt, "show", lambda: None)
    labels = torch.tensor([[1., 0.], [.25, .75]])
    graphs = [Data(pos=torch.tensor([[0., 0.], [1., 0.]]), num_nodes=2,
                   labels=labels.clone(),
                   cells=[box(-.2, -.2, .2, .2), box(.8, -.2, 1.2, .2)]) for _ in range(2)]
    plot_coupling_fgw(*graphs, torch.eye(2)/2, voronoi=True, show_marginals=True,
                     mode="heatmap", label_index=1, labels_names=["A", "B"])
    fig = plt.gcf()
    matrix_ax, alignment = fig.axes[:2]
    top, left = fig.axes[-2:]
    matrix_before = matrix_ax.images[0].get_array().copy()
    original_patches = list(alignment.patches)
    controller = fig._label_heatmap
    assert controller.index == 1
    for index in [0, 1]:
        controller.select(index)
        expected = color_labels(labels, set_colors(2), mode="heatmap", label_index=index).numpy()
        for nodes in [c for c in alignment.collections if isinstance(c, PathCollection)]:
            np.testing.assert_allclose(nodes.get_facecolors()[:, :3], expected)
        np.testing.assert_allclose([p.get_facecolor()[:3] for p in alignment.patches],
                                   np.concatenate([expected, expected]))
        for marginal in [top, left]:
            np.testing.assert_allclose([p.get_facecolor()[:3] for p in marginal.patches], expected)
        assert alignment.get_legend() is None
        colorbar = alignment._label_colorbar
        assert colorbar.ax.get_ylabel() == f"P({['A', 'B'][index]})"
        assert colorbar.mappable.get_clim() == (0, 1)
        np.testing.assert_allclose(colorbar.cmap(0.)[:3], [1., 1., 1.])
        np.testing.assert_allclose(colorbar.cmap(1.)[:3], set_colors(2)[index].numpy())
        assert len(fig.axes) == 5 # Two plots, one colorbar and two marginals.
    assert list(alignment.patches) == original_patches
    np.testing.assert_array_equal(matrix_ax.images[0].get_array(), matrix_before)
    for graph in graphs:
        torch.testing.assert_close(graph.labels, labels)


def test_hungarian_heatmap_shares_one_controller(monkeypatch):
    """SubFigures register on one root controller rather than individual selectors."""
    from matplotlib.collections import PathCollection
    from graphtools.plotting import plot_coupling_hungarian
    monkeypatch.setattr(plt, "show", lambda: None)
    graph = Data(pos=torch.tensor([[0., 0.], [1., 0.]]), labels=torch.eye(2),
                 edge_index=torch.empty((2, 0), dtype=torch.long))
    plot_coupling_hungarian([graph, graph], [graph, graph], [0, 1], mode="heatmap")
    fig = plt.gcf()
    assert len(fig._label_heatmap.updates) == 4
    fig._label_heatmap.select(1)
    expected = color_labels(graph.labels, set_colors(2), mode="heatmap", label_index=1).numpy()
    for ax in fig.axes:
        for artist in ax.collections:
            if isinstance(artist, PathCollection):
                np.testing.assert_allclose(artist.get_facecolors()[:, :3], expected)


def test_heatmap_without_labels_is_inactive():
    """No labels means no controller and the usual red nodes, regardless of index."""
    graph = Data(pos=torch.tensor([[0., 0.], [1., 0.]]),
                 edge_index=torch.empty((2, 0), dtype=torch.long))
    fig, ax = plt.subplots()
    plot_nodes(graph, ax=ax, mode="heatmap", label_index=50)
    assert not hasattr(fig, "_label_heatmap")
    np.testing.assert_allclose(ax.collections[-1].get_facecolors(), [[1., 0., 0., 1.]])


def test_notebook_dropdown_changes_selection(monkeypatch):
    """The notebook creates one dropdown whose changes recolor the existing figure."""
    widgets = pytest.importorskip("ipywidgets")
    import IPython
    import IPython.display
    from types import SimpleNamespace
    from graphtools.plotting.interactive import show_figure
    graph = Data(pos=torch.tensor([[0., 0.], [1., 0.]]), labels=torch.eye(2),
                 edge_index=torch.empty((2, 0), dtype=torch.long))
    fig, ax = plt.subplots()
    plot_nodes(graph, ax=ax, mode="heatmap", labels_names=["A", "B"])
    displayed = []
    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(
        kernel=SimpleNamespace(get_parent=lambda: {})))
    monkeypatch.setattr(IPython.display, "display", lambda obj: displayed.append(obj))
    monkeypatch.setattr(IPython.display, "clear_output", lambda **kwargs: None)
    show_figure(fig)
    panel = displayed[0]
    assert isinstance(panel, widgets.VBox)
    dropdown = panel.children[0]
    assert dropdown.options == (("A", 0), ("B", 1))
    dropdown.value = 1
    assert fig._label_heatmap.index == 1
    assert displayed[-1] is fig
    expected = color_labels(graph.labels, set_colors(2), mode="heatmap", label_index=1)
    np.testing.assert_allclose(ax.collections[-1].get_facecolors()[:, :3], expected.numpy())


@pytest.mark.parametrize("mode", ["standard", "heatmap"])
def test_fgw_hide_legends_preserves_colors(monkeypatch, mode):
    """Suppress category legends/colorbars, while keeping heatmap selection active."""
    from graphtools.plotting import plot_coupling_fgw
    monkeypatch.setattr(plt, "show", lambda: None)
    graph = Data(pos=torch.tensor([[0., 0.], [1., 0.]]), labels=torch.eye(2))
    plot_coupling_fgw(graph, graph, torch.eye(2)/2, mode=mode,
                     show_legend=False, show_marginals=True)
    fig = plt.gcf()
    assert len(fig.axes) == 4 # Heatmap, alignment and two marginal axes only.
    assert all(ax.get_legend() is None for ax in fig.axes)
    if mode == "heatmap":
        fig._label_heatmap.select(1)
        assert len(fig.axes) == 4
        assert not hasattr(fig.axes[1], "_label_colorbar")


@pytest.mark.parametrize("mode", ["standard", "heatmap"])
def test_hungarian_hide_legends(monkeypatch, mode):
    """The grid forwards the legend flag to all of its graph subplots."""
    from graphtools.plotting import plot_coupling_hungarian
    monkeypatch.setattr(plt, "show", lambda: None)
    graph = Data(pos=torch.tensor([[0., 0.], [1., 0.]]), labels=torch.eye(2),
                 edge_index=torch.empty((2, 0), dtype=torch.long))
    plot_coupling_hungarian([graph], [graph], [0], mode=mode, show_legend=False)
    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert all(ax.get_legend() is None for ax in fig.axes)
    if mode == "heatmap":
        fig._label_heatmap.select(1)
        assert len(fig.axes) == 2

"""One category selector per figure, updating artists without rebuilding plots."""
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize


class _LabelHeatmap:
    """Keep node, cell and marginal callbacks synchronized across subplots."""

    def __init__(self, figure, names, index):
        self.figure = figure
        self.names = names
        self.index = index
        self.updates = []

    def register(self, update):
        self.updates.append(update)
        update(self.index)

    def select(self, index):
        if not isinstance(index, int) or not 0 <= index < len(self.names):
            raise ValueError("label_index is outside the available categories")
        self.index = index
        for update in self.updates:
            update(index)
        self.figure.canvas.draw_idle()


def label_controller(figure, count, names, index):
    """Return the shared controller on the root Figure, including SubFigures."""
    while figure.figure is not None and figure.figure is not figure:
        figure = figure.figure
    names = list(names) if names is not None else [str(k) for k in range(count)]
    controller = getattr(figure, "_label_heatmap", None)
    if controller is None:
        controller = _LabelHeatmap(figure, names, index)
        figure._label_heatmap = controller
    elif controller.names != names:
        raise ValueError("All labelled plots in a heatmap figure must share category names and order")
    return controller


def category_colorbar(ax, color, name):
    """Create one probability colorbar per axes and update it in place."""
    previous = ax.get_legend()
    if previous is not None:
        previous.remove()
    cmap = LinearSegmentedColormap.from_list("label_probability", ["white", color])
    colorbar = getattr(ax, "_label_colorbar", None)
    if colorbar is None:
        mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        colorbar = ax.figure.colorbar(mappable, ax=ax, location="right",
                                     fraction=0.046, pad=0.04)
        ax._label_colorbar = colorbar
    else:
        # Reuse the existing axes so changing category never adds another bar.
        colorbar.mappable.set_cmap(cmap)
        colorbar.update_normal(colorbar.mappable)
    colorbar.set_label(f"Weight membership class {name}")


def show_figure(figure):
    """Show a static figure in scripts, or a dropdown and output in notebooks.

    Widgets are imported only for heatmap figures in an IPython kernel.
    The inline backend is supported: redraw the same Figure in one Output,
    without recomputing geometry. Static exports use the selected colors.
    """
    controller = getattr(figure, "_label_heatmap", None)
    if controller is None:
        plt.show()
        return
    try:
        from IPython import get_ipython
        if getattr(get_ipython(), "kernel", None) is None:
            plt.show()
            return
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        plt.show()
        return

    dropdown = widgets.Dropdown(options=[(name, k) for k, name in enumerate(controller.names)],
                                value=controller.index, description="Category:")
    output = widgets.Output()

    def render():
        with output:
            clear_output(wait=True)
            display(figure)

    def changed(change):
        controller.select(change["new"])
        render()

    dropdown.observe(changed, names="value")
    # Prevent the inline backend from displaying a second, static copy.
    plt.close(figure)
    display(widgets.VBox([dropdown, output]))
    render()


def bind_colors(figure, labels, colors, names, index, apply):
    """Register an artist update using a local snapshot of validated labels."""
    from .colors import heatmap_colors
    probabilities = labels.detach().to(device="cpu", dtype=colors.dtype).clone()
    pure_colors = colors.detach().cpu().clone()
    controller = label_controller(figure, len(pure_colors), names, index)

    def update(selected):
        rgb = heatmap_colors(probabilities, pure_colors, selected).numpy()
        apply(rgb, selected)

    controller.register(update)

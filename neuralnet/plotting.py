from typing import Iterable, TypedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mplhep as hep

plt.style.use(hep.style.ATLAS)


class PlotMetricComparison(TypedDict):
    label: str
    x: Iterable[float]
    y: Iterable[float]
    yerr: Iterable[float] | None
    color: str
    marker: str


def plot_metric_comparison(
    metric_label: str,
    ref_name: str,
    to_plot: dict[str, PlotMetricComparison],
    fig_kwargs: dict | plt.Figure = dict(),
    xlim: tuple[float, float] | None = None,
    xticks: Iterable[float] | None = None,
    title: str | None = None,
    scatter_ylim: tuple[float, float] | None = None,
    ref_ratio_ylim: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig = plt.figure(**fig_kwargs)
    grid_spec = GridSpec(9, 1, figure=fig)
    ax = fig.add_subplot(grid_spec[:6, 0])
    ref_ax = fig.add_subplot(grid_spec[6:, 0])
    ax.grid(linestyle="--", alpha=0.1, color="k")
    ax.set_ylabel(metric_label, fontsize="medium")
    ax.set_xticks(xticks)
    ax.set_title(title)
    ax.tick_params(
        axis="x",  # Apply to x and y axes
        which="both",  # Target major ticks
        # bottom=False,
        labelbottom=False,
    )
    ax.tick_params(
        axis="y",  # Apply to x and y axes
        which="both",  # Target major ticks
        # bottom=False,
        labelsize="small",
    )

    ref_ax.grid(linestyle="--", alpha=0.1, color="k")
    ref_ax.set_xticks(xticks)
    ref_ax.tick_params(
        axis="y",  # Apply to x and y axes
        which="both",  # Target major ticks
        # bottom=False,
        labelsize="small",
    )
    ref_ax.set_xlabel("Fold", fontsize="medium")
    ref_ax.tick_params(
        axis="x",  # Apply to x and y axes
        which="both",  # Target major ticks
        # bottom=False,
        labelsize="small",
    )
    ref_ax.set_ylabel(f"Model/Ref ({to_plot[ref_name]['label']})", fontsize="medium")
    ref_ax.axhline(1, color="k", linestyle="--", alpha=0.5)

    if xlim is not None:
        ax.set_xlim(*xlim)
        ref_ax.set_xlim(*xlim)

    if scatter_ylim is not None:
        ax.set_ylim(*scatter_ylim)

    if ref_ratio_ylim is not None:
        ref_ax.set_ylim(*ref_ratio_ylim)

    for plot_group in to_plot.values():
        x = plot_group["x"]
        y = plot_group["y"]
        yerr = plot_group.get("yerr", None)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=plot_group["label"],
            color=plot_group["color"],
            fmt="o",
            marker=plot_group["marker"],
            markeredgecolor="k",
            markersize=10,
            alpha=0.7,
        )

    i = 0
    for group_name, plot_group in to_plot.items():
        if group_name == ref_name:
            continue
        x = plot_group["x"]
        m = plot_group["y"]
        r = to_plot[ref_name]["y"]
        m_err = plot_group["yerr"]
        r_err = to_plot[ref_name]["yerr"]
        z = m / r
        z_err = np.abs(z) * np.sqrt((m_err / m) ** 2 + (r_err / r) ** 2)
        ref_ax.errorbar(
            x,
            z,
            yerr=z_err,
            label=plot_group["label"],
            color=plot_group["color"],
            fmt="o",
            marker=plot_group["marker"],
            markeredgecolor="k",
            markersize=10,
            alpha=0.7,
        )
        i += 1
    ax.legend(fontsize="small")
    # ref_ax.legend(fontsize='small')
    fig.tight_layout()
    return fig, ax, ref_ax

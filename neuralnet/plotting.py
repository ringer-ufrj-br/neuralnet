from typing import Iterable, TypedDict, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mplhep as hep
import polars as pl

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
    top_ax = fig.add_subplot(grid_spec[:6, 0])
    bottom_ax = fig.add_subplot(grid_spec[6:, 0])
    top_ax.grid(linestyle="--", alpha=0.1, color="k")
    top_ax.set_ylabel(metric_label, fontsize="medium")
    top_ax.set_xticks(xticks)
    top_ax.set_title(title)
    top_ax.tick_params(
        axis="x",  # Apply to x and y axes
        which="both",  # Target major ticks
        # bottom=False,
        labelbottom=False,
    )
    top_ax.tick_params(
        axis="y",  # Apply to x and y axes
        which="both",  # Target major ticks
        # bottom=False,
        labelsize="small",
    )

    bottom_ax.grid(linestyle="--", alpha=0.1, color="k")
    bottom_ax.set_xticks(xticks)
    bottom_ax.tick_params(
        axis="y",  # Apply to x and y axes
        which="both",  # Target major ticks
        # bottom=False,
        labelsize="small",
    )
    bottom_ax.set_xlabel("Fold", fontsize="medium")
    bottom_ax.tick_params(
        axis="x",  # Apply to x and y axes
        which="both",  # Target major ticks
        # bottom=False,
        labelsize="small",
    )
    bottom_ax.set_ylabel(
        rf"$\frac{{\mathrm{{Model}}}}{{{to_plot[ref_name]['label']}}}$",
        fontsize="medium",
        rotation=90,
    )
    bottom_ax.axhline(1, color="k", linestyle="--", alpha=0.5, label="Equal")

    if xlim is not None:
        top_ax.set_xlim(*xlim)
        bottom_ax.set_xlim(*xlim)

    if scatter_ylim is not None:
        top_ax.set_ylim(*scatter_ylim)

    if ref_ratio_ylim is not None:
        bottom_ax.set_ylim(*ref_ratio_ylim)

    for plot_group in to_plot.values():
        x = plot_group["x"]
        y = plot_group["y"]
        yerr = plot_group.get("yerr", None)
        top_ax.errorbar(
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
        bottom_ax.errorbar(
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
    top_ax.legend(fontsize="small")
    bottom_ax.legend(fontsize="small")
    fig.tight_layout()
    return fig, top_ax, bottom_ax


class VariableDistributionComparisonDict(TypedDict):
    name: str
    label: str
    color: str


def plot_variable_distribution_comparison(
    variable_label: str,
    ref: VariableDistributionComparisonDict,
    to_compare: list[VariableDistributionComparisonDict],
    data: pl.DataFrame | pl.LazyFrame,
    fig_kwargs: dict | plt.Figure | None = None,
    hist_kwargs: dict | None = None,
    title: str | None = None,
    scale: Literal["linear", "log"] = "linear",
    top_legend_kwargs: dict | None = None,
    bottom_legend_kwargs: dict | None = None,
    bottom_ax_set: dict | None = None,
    scatter: bool = False,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    if fig_kwargs is None:
        fig = plt.figure()
    elif isinstance(fig_kwargs, plt.Figure):
        fig = fig_kwargs
    elif isinstance(fig_kwargs, dict):
        fig = plt.figure(**fig_kwargs)
    else:
        raise TypeError("fig_kwargs must be a dict, plt.Figure, or None")

    if hist_kwargs is None:
        hist_kwargs = dict(alpha=0.5, bins=50, density=True)
    elif isinstance(hist_kwargs, dict):
        hist_kwargs["density"] = True
    else:
        raise TypeError("hist_kwargs must be a dict or None")

    if top_legend_kwargs is None:
        top_legend_kwargs = dict(loc="best")
    elif isinstance(top_legend_kwargs, dict):
        pass
    else:
        raise TypeError("top_legend_kwargs must be a dict or None")

    if bottom_legend_kwargs is None:
        bottom_legend_kwargs = dict(loc="best")
    elif isinstance(bottom_legend_kwargs, dict):
        pass
    else:
        raise TypeError("bottom_legend_kwargs must be a dict or None")

    if isinstance(data, pl.LazyFrame):
        data = data.select(ref["name"], *[var["name"] for var in to_compare]).collect()

    grid_spec = GridSpec(9, 1, figure=fig)
    top_ax = fig.add_subplot(grid_spec[:6, 0])
    bottom_ax = fig.add_subplot(grid_spec[6:, 0])
    ref_heights, ref_bins, _ = top_ax.hist(
        data[ref["name"]].to_numpy(),
        label=ref["label"] if not scatter else None,
        color=ref["color"],
        histtype="step",
        **hist_kwargs,
    )
    if scatter:
        ref_scatter_x = (ref_bins[:-1] + ref_bins[1:]) / 2
        top_ax.scatter(
            ref_scatter_x,
            y=ref_heights,
            label=ref["label"],
            color=ref["color"],
            marker=ref["marker"] if "marker" in ref else "o",
            edgecolor="k",
            alpha=0.5,
        )

    # Removes 'bins' from hist_kwargs to avoid passing it again in the loop below
    if "bins" in hist_kwargs:
        hist_kwargs.pop("bins")

    for var in to_compare:
        heights, _, _ = top_ax.hist(
            data[var["name"]].to_numpy(),
            label=var["label"] if not scatter else None,
            color=var["color"],
            bins=ref_bins,
            histtype="step",
            **hist_kwargs,
        )
        if scatter:
            top_ax.scatter(
                ref_scatter_x,
                y=heights,
                label=var["label"],
                color=var["color"],
                marker=var["marker"] if "marker" in var else "o",
                edgecolor="k",
                alpha=0.5,
            )
        rel_diff = ((heights - ref_heights) / ref_heights) * 100
        bottom_ax.bar(
            x=ref_bins[:-1],
            align="edge",
            height=rel_diff,
            width=np.diff(ref_bins),
            alpha=0.5,
            color="none",
            edgecolor=var["color"],
            label=var["label"],
        )
    top_ax.grid(linestyle="--", alpha=0.1, color="k")
    top_ax.set_ylabel("Density", fontsize="medium")
    top_ax.set_yscale(scale)
    top_ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
    )
    top_ax.tick_params(
        axis="y",
        which="both",
        labelsize="small",
    )
    if title is not None:
        top_ax.set_title(title, fontsize="large")
    bottom_ax.grid(linestyle="--", alpha=0.1, color="k")
    bottom_ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    bottom_ax.set_xlabel(variable_label, fontsize="medium")
    bottom_ax.set_ylabel(f"Diff to {ref['label']} (%)", fontsize="medium")
    bottom_ax.tick_params(
        axis="x",
        which="both",
        labelsize="small",
    )
    bottom_ax.tick_params(
        axis="y",
        which="both",
        labelsize="small",
    )
    if bottom_ax_set is not None:
        bottom_ax.set(**bottom_ax_set)

    top_ax.legend(**top_legend_kwargs)
    bottom_ax.legend(**bottom_legend_kwargs)

    return fig, top_ax, bottom_ax


def plot_roc_curve(
    data: pl.LazyFrame | pl.DataFrame,
    tpr_col: str,
    fpr_col: str,
    threshold_fit_results: dict,
    references: dict,
    et_bin: dict,
    eta_bin: dict,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:

    fig, ax = plt.subplots()
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    fpr = data[fpr_col].to_numpy()
    tpr = data[tpr_col].to_numpy()
    ax.grid(linestyle="--", alpha=0.1, color="k")
    ax.plot(fpr, tpr, label="ROC Curve", color="tab:blue")

    ax_inset = ax.inset_axes([0.45, 0.3, 0.5, 0.45])
    # Plot ROC curve and elbow on the inset axis
    ax_inset.plot(fpr, tpr, color="tab:blue")
    # ax_inset.plot(elbow_fpr, elbow_tpr, 'ro', markersize=6)
    ax_inset.grid(linestyle="--", alpha=0.3)
    ref_tprs = []
    ref_fprs = []

    for ref, ref_info in references.items():
        ref_result = threshold_fit_results[ref]
        ref_tpr = ref_result["tpr"]
        ref_tprs.append(ref_tpr)
        ref_fpr = ref_result["fpr"]
        ref_fprs.append(ref_fpr)
        ax.plot(
            ref_fpr,
            ref_tpr,
            "ro",
            color=ref_info["color"],
            label=f"{ref_info['label']} (FPR={ref_fpr * 100:.2f}%, TPR={ref_tpr * 100:.2f}%)",
        )
        ax_inset.plot(ref_fpr, ref_tpr, "ro", markersize=6, color=ref_info["color"])

    # Set zoom limits centered around the operation points
    ax_inset.set_xlim(min(ref_fprs) * 0.9, max(ref_fprs) * 1.05)
    ax_inset.set_ylim(min(ref_tprs) * 0.9, max(ref_tprs) * 1.05)
    # Draw indicator box and connecting lines from the main axis to the inset
    ax.indicate_inset_zoom(ax_inset, edgecolor="red", alpha=0.7)
    ax.legend(loc="lower left")
    et_high = et_bin["high"]
    if et_high == "Infinity":
        et_high_label = "\\infty"
    else:
        et_high_label = int(et_high * 1e-3)
    ax.set_title(
        f"$E_T \\in [{int(et_bin['low'] * 1e-3)}, {et_high_label})$ and $|\\eta| \\in [{eta_bin['low'] * 1e-3:.2f}, {eta_bin['high'] * 1e-3:.2f})$"
    )
    ax.set_xlabel("Fake Postive Rate")
    ax.set_ylabel("True Positve Rate")
    fig.tight_layout()
    return fig, ax, ax_inset

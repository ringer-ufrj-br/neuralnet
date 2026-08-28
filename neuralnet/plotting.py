from collections.abc import Iterable
from typing import TypedDict, Literal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize
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


DEFAULT_METRIC_LABELS: dict[str, str] = {
    "mse": "Mean Squared Error (MSE)",
    "rmse": "Root Mean Squared Error (RMSE)",
    "mae": "Mean Absolute Error (MAE)",
    "mape": "Mean Absolute Percentage Error (MAPE)",
    "max_error": "Maximum Absolute Error",
    "kl_divergence": "Spatial KL Divergence ($D_{\\mathrm{KL}}$)",
    "hist_kl_divergence": "Histogram KL Divergence ($D_{\\mathrm{KL}}$)",
}


def plot_metrics_heatmap(
    data: pl.DataFrame | pl.LazyFrame,
    metric_col: str = "mse",
    x_col: str = "fractional_bits",
    y_col: str = "integer_bits",
    title: str | None = None,
    metric_label: str | None = None,
    ax: plt.Axes | None = None,
    fig_kwargs: dict | None = None,
    cmap: str = "viridis_r",
    log_scale: bool = True,
    annot: bool = True,
    fmt: str = ".2e",
    cbar: bool = True,
    cbar_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 2D heatmap of an aggregated metric across integer and fractional bit widths.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        DataFrame containing quantization results with integer_bits, fractional_bits, and metric columns.
    metric_col : str, default="mse"
        Column name of the metric to plot.
    x_col : str, default="fractional_bits"
        Column name for horizontal axis.
    y_col : str, default="integer_bits"
        Column name for vertical axis.
    title : str | None, optional
        Title of the plot.
    metric_label : str | None, optional
        Colorbar and metric label. If None, looks up from DEFAULT_METRIC_LABELS.
    ax : plt.Axes | None, optional
        Existing Axes to draw into. If None, creates a new figure and axes.
    fig_kwargs : dict | None, optional
        Keyword arguments for plt.figure when ax is None.
    cmap : str, default="viridis_r"
        Colormap name.
    log_scale : bool, default=True
        Whether to use LogNorm for color scaling when values are positive.
    annot : bool, default=True
        Whether to annotate each heatmap cell with numerical value.
    fmt : str, default=".2e"
        Format string for cell text annotations.
    cbar : bool, default=True
        Whether to display colorbar.
    cbar_kwargs : dict | None, optional
        Additional kwargs for colorbar.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and Axes objects.
    """
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    if ax is None:
        fig = plt.figure(**(fig_kwargs or {}))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    x_vals = sorted(data[x_col].unique().to_list())
    y_vals = sorted(data[y_col].unique().to_list())

    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    x_idx_map = {v: i for i, v in enumerate(x_vals)}
    y_idx_map = {v: i for i, v in enumerate(y_vals)}

    for row in data.iter_rows(named=True):
        if row[y_col] in y_idx_map and row[x_col] in x_idx_map:
            val = row[metric_col]
            if val is not None:
                grid[y_idx_map[row[y_col]], x_idx_map[row[x_col]]] = float(val)

    valid_vals = grid[~np.isnan(grid)]
    min_val = float(np.min(valid_vals)) if len(valid_vals) > 0 else 1e-12
    max_val = float(np.max(valid_vals)) if len(valid_vals) > 0 else 1.0

    if log_scale and min_val > 0 and max_val > min_val:
        norm = LogNorm(vmin=min_val, vmax=max_val)
    else:
        norm = Normalize(vmin=min_val, vmax=max_val)

    im = ax.imshow(grid, origin="lower", cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals)

    ax.set_xlabel("Fractional Bits ($fb$)", fontsize="medium")
    ax.set_ylabel("Integer Bits ($ib$)", fontsize="medium")

    label = metric_label or DEFAULT_METRIC_LABELS.get(metric_col, metric_col)
    if title is not None:
        ax.set_title(title, fontsize="large")
    else:
        ax.set_title(label, fontsize="large")

    if annot:
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                val = grid[i, j]
                if not np.isnan(val):
                    # Pick contrasting text color based on normalized value
                    normed_val = norm(val)
                    text_color = "white" if normed_val < 0.4 else "black"
                    text = f"{val:{fmt}}" if fmt else f"{val}"
                    ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize="small", fontweight="bold")

    if cbar:
        cb_kwargs = cbar_kwargs or {}
        cbar_obj = fig.colorbar(im, ax=ax, **cb_kwargs)
        cbar_obj.set_label(label, fontsize="medium")

    return fig, ax


def plot_aggregated_metrics_grid(
    data: pl.DataFrame | pl.LazyFrame,
    metrics: list[str] | None = None,
    x_col: str = "fractional_bits",
    y_col: str = "integer_bits",
    metric_labels: dict[str, str] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis_r",
    log_scale: bool = True,
    annot: bool = True,
    fmt: str = ".2e",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a grid of heatmaps for multiple aggregated quantization metrics.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        Results DataFrame from AlternativeNorm1Analysis.
    metrics : list[str] | None, optional
        List of metric column names. Defaults to standard error and divergence metrics.
    x_col : str, default="fractional_bits"
        Column name for x-axis.
    y_col : str, default="integer_bits"
        Column name for y-axis.
    metric_labels : dict[str, str] | None, optional
        Custom dictionary mapping metric column name to display label.
    ncols : int, default=3
        Number of columns in subplot grid.
    figsize : tuple[float, float] | None, optional
        Figure size.
    cmap : str, default="viridis_r"
        Colormap name.
    log_scale : bool, default=True
        Whether to use log scale for colormap.
    annot : bool, default=True
        Whether to annotate cells with values.
    fmt : str, default=".2e"
        Number formatting for cell annotations.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and array of Axes.
    """
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    all_available = [
        col
        for col in ["mse", "rmse", "mae", "mape", "max_error", "kl_divergence", "hist_kl_divergence"]
        if col in data.columns
    ]
    target_metrics = metrics or all_available
    if not target_metrics:
        raise ValueError("No metrics found to plot in data.")

    n_metrics = len(target_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    if figsize is None:
        figsize = (6.0 * ncols, 5.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for idx, metric in enumerate(target_metrics):
        ax = flat_axes[idx]
        lbl = (metric_labels or {}).get(metric, DEFAULT_METRIC_LABELS.get(metric, metric))
        plot_metrics_heatmap(
            data=data,
            metric_col=metric,
            x_col=x_col,
            y_col=y_col,
            metric_label=lbl,
            title=lbl,
            ax=ax,
            cmap=cmap,
            log_scale=log_scale,
            annot=annot,
            fmt=fmt,
            cbar=True,
        )

    # Hide unused subplots
    for idx in range(n_metrics, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_joint_hist2d_with_marginals(
    x: np.ndarray | Iterable[float] | None = None,
    y: np.ndarray | Iterable[float] | None = None,
    data: pl.DataFrame | pl.LazyFrame | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str | None = None,
    bins: int | tuple[int, int] = 50,
    cmap: str = "Blues",
    log_counts: bool = True,
    xscale: Literal["linear", "log"] = "linear",
    yscale: Literal["linear", "log"] = "linear",
    marginal_color: str = "tab:blue",
    marginal_density: bool = True,
    fig_kwargs: dict | plt.Figure | None = None,
    cbar: bool = True,
    cbar_label: str = "Counts",
) -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    """Plot a 2D histogram with marginal 1D distributions on top and right axes.

    Parameters
    ----------
    x : np.ndarray | Iterable[float] | None, optional
        Data for horizontal axis (if data / x_col not provided).
    y : np.ndarray | Iterable[float] | None, optional
        Data for vertical axis (if data / y_col not provided).
    data : pl.DataFrame | pl.LazyFrame | None, optional
        Optional DataFrame containing x_col and y_col.
    x_col : str | None, optional
        Column name in data for x-axis.
    y_col : str | None, optional
        Column name in data for y-axis.
    xlabel : str, default="X"
        Label for horizontal axis.
    ylabel : str, default="Y"
        Label for vertical axis.
    title : str | None, optional
        Title above top marginal distribution.
    bins : int | tuple[int, int], default=50
        Number of bins for 2D histogram and marginals.
    cmap : str, default="Blues"
        Colormap for 2D histogram.
    log_counts : bool, default=True
        Whether to use LogNorm for 2D histogram counts.
    xscale : Literal["linear", "log"], default="linear"
        Scale for horizontal axis.
    yscale : Literal["linear", "log"], default="linear"
        Scale for vertical axis.
    marginal_color : str, default="tab:blue"
        Fill and edge color for marginal histograms.
    marginal_density : bool, default=True
        Whether marginal histograms show density (normalized area = 1).
    fig_kwargs : dict | plt.Figure | None, optional
        Figure settings.
    cbar : bool, default=True
        Whether to draw colorbar.
    cbar_label : str, default="Counts"
        Colorbar label.

    Returns
    -------
    tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]
        Tuple of (Figure, Main 2D Axes, Top Marginal Axes, Right Marginal Axes).
    """
    if data is not None:
        if isinstance(data, pl.LazyFrame):
            data = data.select(x_col, y_col).collect()
        x = data[x_col].to_numpy()
        y = data[y_col].to_numpy()
        if xlabel == "X" and x_col is not None:
            xlabel = x_col
        if ylabel == "Y" and y_col is not None:
            ylabel = y_col
    else:
        if x is None or y is None:
            raise ValueError("Must provide either (data, x_col, y_col) or (x, y).")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

    # Filter out NaNs / Infs
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]

    if isinstance(fig_kwargs, plt.Figure):
        fig = fig_kwargs
    elif isinstance(fig_kwargs, dict):
        fig = plt.figure(**fig_kwargs)
    else:
        fig = plt.figure(figsize=(8, 7))

    grid_spec = GridSpec(7, 7, figure=fig, hspace=0.15, wspace=0.15)
    top_ax = fig.add_subplot(grid_spec[0:2, 0:5])
    main_ax = fig.add_subplot(grid_spec[2:7, 0:5])
    right_ax = fig.add_subplot(grid_spec[2:7, 5:6])
    cbar_ax = fig.add_subplot(grid_spec[2:7, 6:7]) if cbar else None

    # Handle log scale filtering for bins
    if xscale == "log":
        valid_x = x > 0
        x = x[valid_x]
        y = y[valid_x]
    if yscale == "log":
        valid_y = y > 0
        x = x[valid_y]
        y = y[valid_y]

    norm = LogNorm() if log_counts else Normalize()
    cmin_val = 1 if log_counts else None

    # 2D Histogram
    _h2d, xedges, yedges, im = main_ax.hist2d(x, y, bins=bins, cmap=cmap, norm=norm, cmin=cmin_val)

    main_ax.set_xlabel(xlabel, fontsize="medium")
    main_ax.set_ylabel(ylabel, fontsize="medium")
    main_ax.set_xscale(xscale)
    main_ax.set_yscale(yscale)
    main_ax.grid(linestyle="--", alpha=0.2, color="k")

    # Top marginal histogram
    top_ax.hist(
        x,
        bins=xedges,
        density=marginal_density,
        color=marginal_color,
        edgecolor=marginal_color,
        alpha=0.6,
        histtype="stepfilled",
    )
    top_ax.set_xlim(main_ax.get_xlim())
    top_ax.set_xscale(xscale)
    top_ax.set_ylabel("Density" if marginal_density else "Counts", fontsize="small")
    top_ax.tick_params(axis="x", labelbottom=False)
    top_ax.grid(linestyle="--", alpha=0.2, color="k")
    if title is not None:
        top_ax.set_title(title, fontsize="large")

    # Right marginal histogram
    right_ax.hist(
        y,
        bins=yedges,
        density=marginal_density,
        orientation="horizontal",
        color=marginal_color,
        edgecolor=marginal_color,
        alpha=0.6,
        histtype="stepfilled",
    )
    right_ax.set_ylim(main_ax.get_ylim())
    right_ax.set_yscale(yscale)
    right_ax.set_xlabel("Density" if marginal_density else "Counts", fontsize="small")
    right_ax.tick_params(axis="y", labelleft=False)
    right_ax.grid(linestyle="--", alpha=0.2, color="k")

    # Colorbar
    if cbar and cbar_ax is not None and im is not None:
        fig.colorbar(im, cax=cbar_ax, label=cbar_label)

    return fig, main_ax, top_ax, right_ax


def plot_error_distributions_by_bits(
    diff_dfs: dict[tuple[int, int], pl.DataFrame | pl.LazyFrame],
    metric_col: str = "sample_mae",
    fixed_bit_type: Literal["integer_bits", "fractional_bits"] = "integer_bits",
    fixed_bit_val: int = 1,
    xlabel: str | None = None,
    title: str | None = None,
    bins: int = 50,
    log_x: bool = True,
    log_y: bool = True,
    density: bool = True,
    ax: plt.Axes | None = None,
    fig_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot overlaid 1D marginal error distributions across varying bit settings.

    Parameters
    ----------
    diff_dfs : dict[tuple[int, int], pl.DataFrame | pl.LazyFrame]
        Dictionary mapping (ib, fb) tuples to sample differences DataFrames.
    metric_col : str, default="sample_mae"
        Sample-level error metric column to plot.
    fixed_bit_type : Literal["integer_bits", "fractional_bits"], default="integer_bits"
        Which bit width to keep fixed while varying the other.
    fixed_bit_val : int, default=1
        Value of the fixed bit width.
    xlabel : str | None, optional
        Label for horizontal axis.
    title : str | None, optional
        Title of the plot.
    bins : int, default=50
        Number of histogram bins.
    log_x : bool, default=True
        Whether to use log scale for x-axis.
    log_y : bool, default=True
        Whether to use log scale for y-axis.
    density : bool, default=True
        Whether to normalize histogram as probability density.
    ax : plt.Axes | None, optional
        Existing Axes to draw into.
    fig_kwargs : dict | None, optional
        Keyword arguments for plt.figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and Axes.
    """
    if ax is None:
        fig = plt.figure(**(fig_kwargs or {}))
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Filter matching configurations
    filtered_items = []
    for (ib, fb), df in diff_dfs.items():
        if fixed_bit_type == "integer_bits" and ib == fixed_bit_val:
            filtered_items.append((fb, f"fb={fb} (ib={ib})", df))
        elif fixed_bit_type == "fractional_bits" and fb == fixed_bit_val:
            filtered_items.append((ib, f"ib={ib} (fb={fb})", df))

    filtered_items.sort(key=lambda x: x[0])
    if not filtered_items:
        raise ValueError(f"No configurations found with {fixed_bit_type}={fixed_bit_val}")

    all_vals = []
    collected_dfs = []
    for val_idx, label_str, df in filtered_items:
        c_df = df.collect() if isinstance(df, pl.LazyFrame) else df
        arr = c_df[metric_col].to_numpy()
        arr = arr[np.isfinite(arr)]
        if log_x:
            arr = arr[arr > 0]
        all_vals.append(arr)
        collected_dfs.append((val_idx, label_str, arr))

    flat = np.concatenate(all_vals) if all_vals else np.array([1e-6, 1.0])
    if len(flat) == 0:
        flat = np.array([1e-6, 1.0])

    if log_x:
        min_v = max(float(np.min(flat)), 1e-12)
        max_v = float(np.max(flat))
        bin_edges = np.logspace(np.log10(min_v), np.log10(max_v), bins)
    else:
        bin_edges = np.linspace(float(np.min(flat)), float(np.max(flat)), bins)

    for i, (val_idx, label_str, arr) in enumerate(collected_dfs):
        color = color_cycle[i % len(color_cycle)]
        ax.hist(
            arr,
            bins=bin_edges,
            density=density,
            histtype="step",
            linewidth=2.0,
            label=label_str,
            color=color,
        )

    ax.grid(linestyle="--", alpha=0.2, color="k")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    label = xlabel or metric_col
    ax.set_xlabel(label, fontsize="medium")
    ax.set_ylabel("Density" if density else "Counts", fontsize="medium")

    plot_title = title or f"Error Distribution ({fixed_bit_type} = {fixed_bit_val})"
    ax.set_title(plot_title, fontsize="large")
    ax.legend(fontsize="small", loc="best")

    return fig, ax


class QuadrantModelConfig(TypedDict, total=False):
    """Configuration for a model prediction column in :func:`quadrant_plot`.

    Attributes
    ----------
    col : str
        Column name in the DataFrame containing Boolean predictions for this model.
    label : str
        Display label for the model (used in quadrant legend labels).
    """

    col: str
    label: str


class QuadrantVariableConfig(TypedDict, total=False):
    """Configuration for a variable column in :func:`quadrant_plot`.

    Attributes
    ----------
    col : str
        Column name in the DataFrame containing the continuous variable to plot.
    label : str
        Display label for the variable (used for the x-axis label).
    """

    col: str
    label: str


class QuadrantHistConfig(TypedDict, total=False):
    """Per-quadrant visual configuration for :func:`quadrant_plot`.

    All fields are optional; sensible defaults are applied for any key that is
    omitted.

    Attributes
    ----------
    color : str
        Matplotlib color string used for the histogram step line and optional
        scatter markers.
    marker : str
        Matplotlib marker string.  Used only when *scatter* is ``True``.
    markersize : float
        Marker size in points**2 (passed to ``s`` in :func:`matplotlib.axes.Axes.scatter`).
        Used only when *scatter* is ``True``.
    alpha : float
        Opacity for the histogram step line and scatter markers.
    label : str
        Legend label override.  If omitted a default label with the sample
        count is used.
    hist_kwargs : dict
        Extra keyword arguments forwarded to :func:`matplotlib.axes.Axes.hist`
        for this quadrant (e.g. ``linewidth``).
    """

    color: str
    marker: str
    markersize: float
    alpha: float
    label: str
    hist_kwargs: dict


# Default visual config for each quadrant (overridable by the caller).
_QUADRANT_DEFAULTS: dict[str, QuadrantHistConfig] = {
    "both_false": {"color": "black", "marker": "o", 'alpha': 0.3},
    "both_true": {"color": "black", "marker": "s"},
    "a_only": {"color": "blue", "marker": "^"},
    "b_only": {"color": "red", "marker": "v"},
}


def quadrant_plot(
    data: pl.DataFrame | pl.LazyFrame,
    model_a_col: QuadrantModelConfig | str,
    model_b_col: QuadrantModelConfig | str,
    variable_col: QuadrantVariableConfig | str,
    variable_label: str | None = None,
    bins: int | np.ndarray = 50,
    density: bool = True,
    scatter: bool = False,
    quadrant_config: dict[str, QuadrantHistConfig] | None = None,
    fig_kwargs: dict | None = None,
    hist_kwargs: dict | None = None,
    title: str | None = None,
    legend_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes, dict[str, np.ndarray]]:
    """Overlay histograms of *variable_col* split by the four prediction quadrants.

    The four quadrants are defined by the Boolean outputs of two binary
    classifiers (*model_a_col* and *model_b_col*):

    ============   =============  =============  ================================
    Quadrant key   Model A        Model B        Meaning
    ============   =============  =============  ================================
    ``both_false`` ``False``      ``False``      Both models reject the sample
    ``both_true``  ``True``       ``True``       Both models accept the sample
    ``a_only``     ``True``       ``False``      Only model A accepts
    ``b_only``     ``False``      ``True``       Only model B accepts
    ============   =============  =============  ================================

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        Input data containing at minimum the columns specified in
        *model_a_col*, *model_b_col*, and *variable_col*.
    model_a_col : QuadrantModelConfig | str
        Dictionary containing plot configurations for Model A (e.g.
        ``{"col": "modelA", "label": "Model A"}``) or column name.
    model_b_col : QuadrantModelConfig | str
        Dictionary containing plot configurations for Model B (e.g.
        ``{"col": "modelB", "label": "Model B"}``) or column name.
    variable_col : QuadrantVariableConfig | str
        Dictionary containing plot configurations for the continuous variable (e.g.
        ``{"col": "score", "label": "Score"}``) or column name.
    variable_label : str | None, optional
        X-axis label override. Defaults to label in *variable_col* or column name.
    bins : int | np.ndarray, default=50
        Histogram bin specification.  A shared set of edges is built from all
        non-empty quadrant values so that comparisons across quadrants are
        meaningful.  Pass a ``np.ndarray`` to supply explicit edges.
    density : bool, default=True
        If ``True`` each histogram / marker distribution on the top axis is normalised to unit area.
    scatter : bool, default=False
        If ``True``, plot only the markers at the bin centres without drawing
        the histogram bins.
    quadrant_config : dict[str, QuadrantHistConfig] | None, optional
        Per-quadrant visual overrides keyed by quadrant name
        (``"both_false"``, ``"both_true"``, ``"a_only"``, ``"b_only"``).
        Missing keys or missing inner fields fall back to the defaults
        (including values derived from *model_a_col* and *model_b_col*).  Example::

            quadrant_config = {
                "both_true": {"color": "crimson", "marker": "*", "label": "Both accept"},
            }
    fig_kwargs : dict | None, optional
        Keyword arguments forwarded to :func:`matplotlib.pyplot.figure`.
    hist_kwargs : dict | None, optional
        Shared keyword arguments forwarded to every
        :func:`matplotlib.axes.Axes.hist` call (e.g. ``alpha``,
        ``linewidth``).  Per-quadrant ``hist_kwargs`` inside
        *quadrant_config* take precedence over these.
    title : str | None, optional
        Axes title (placed on top axis).
    legend_kwargs : dict | None, optional
        Keyword arguments forwarded to :func:`matplotlib.axes.Axes.legend`.

    Returns
    -------
    fig : plt.Figure
        The figure containing the subplots.
    top_ax : plt.Axes
        The top axes containing the 4-quadrant distributions.
    bottom_ax : plt.Axes
        The bottom axes containing the percentage of data that only one model approves.
    quadrant_data : dict[str, np.ndarray]
        Mapping from quadrant name to the 1-D float array of *variable_col*
        values that belong to that quadrant.  Useful for downstream analysis.
    """
    # ------------------------------------------------------------------
    # Parse model and variable configuration dicts
    # ------------------------------------------------------------------
    if isinstance(model_a_col, str):
        model_a_col = {"col": model_a_col}
    if isinstance(model_b_col, str):
        model_b_col = {"col": model_b_col}
    if isinstance(variable_col, str):
        variable_col = {"col": variable_col}

    a_col_name = model_a_col["col"]
    b_col_name = model_b_col["col"]
    var_col_name = variable_col["col"]

    a_label = model_a_col.get("label", a_col_name)
    b_label = model_b_col.get("label", b_col_name)
    var_label = variable_col.get("label", variable_label if variable_label is not None else var_col_name)

    # ------------------------------------------------------------------
    # Validate scatter
    # ------------------------------------------------------------------
    if not isinstance(scatter, bool):
        raise TypeError("scatter must be a bool.")

    # ------------------------------------------------------------------
    # Collect / validate data
    # ------------------------------------------------------------------
    if isinstance(data, pl.LazyFrame):
        data = data.select(a_col_name, b_col_name, var_col_name).collect()

    a = data[a_col_name]
    b = data[b_col_name]

    if a.dtype != pl.Boolean:
        raise TypeError(f"Column '{a_col_name}' must be of dtype Boolean, got {a.dtype}.")
    if b.dtype != pl.Boolean:
        raise TypeError(f"Column '{b_col_name}' must be of dtype Boolean, got {b.dtype}.")

    var = data[var_col_name].to_numpy().astype(float)
    a_np = a.to_numpy()
    b_np = b.to_numpy()

    quadrant_data: dict[str, np.ndarray] = {
        "both_false": var[~a_np & ~b_np],
        "both_true": var[a_np & b_np],
        "a_only": var[a_np & ~b_np],
        "b_only": var[~a_np & b_np],
    }

    # ------------------------------------------------------------------
    # Create figure and 2-subplot GridSpec layout (3:1 ratio)
    # ------------------------------------------------------------------
    if fig_kwargs is None:
        fig = plt.figure()
    elif isinstance(fig_kwargs, dict):
        fig = plt.figure(**fig_kwargs)
    else:
        raise TypeError("fig_kwargs must be a dict or None.")

    grid_spec = GridSpec(4, 1, figure=fig)
    top_ax = fig.add_subplot(grid_spec[:3, 0])
    bottom_ax = fig.add_subplot(grid_spec[3:, 0], sharex=top_ax)

    # ------------------------------------------------------------------
    # Resolve shared histogram kwargs
    # ------------------------------------------------------------------
    base_hist_kw: dict = {"alpha": 0.6, "histtype": "step", "linewidth": 1.5}
    if hist_kwargs is not None:
        base_hist_kw.update(hist_kwargs)

    # ------------------------------------------------------------------
    # Determine common bin edges from the union of all non-empty arrays
    # ------------------------------------------------------------------
    non_empty = [arr for arr in quadrant_data.values() if len(arr) > 0]
    all_values = np.concatenate(non_empty) if non_empty else np.array([0.0, 1.0])

    if isinstance(bins, np.ndarray):
        bin_edges: np.ndarray = bins
    else:
        bin_edges = np.linspace(float(np.nanmin(all_values)), float(np.nanmax(all_values)), bins + 1)

    # ------------------------------------------------------------------
    # Merge per-quadrant config with defaults
    # ------------------------------------------------------------------
    base_quadrant_defaults: dict[str, QuadrantHistConfig] = {
        "both_false": dict(_QUADRANT_DEFAULTS["both_false"]),
        "both_true": dict(_QUADRANT_DEFAULTS["both_true"]),
        "a_only": dict(_QUADRANT_DEFAULTS["a_only"]),
        "b_only": dict(_QUADRANT_DEFAULTS["b_only"]),
    }

    quadrant_config = quadrant_config or {}
    resolved: dict[str, QuadrantHistConfig] = {}
    for qname, defaults in base_quadrant_defaults.items():
        override = quadrant_config.get(qname, {})
        resolved[qname] = {**defaults, **override}  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Plot each quadrant on top_ax
    # ------------------------------------------------------------------
    default_labels: dict[str, str] = {
        "both_false": f"Both False (N={len(quadrant_data['both_false'])})",
        "both_true": f"Both True (N={len(quadrant_data['both_true'])})",
        "a_only": f"{a_label} approved (N={len(quadrant_data['a_only'])})",
        "b_only": f"{b_label} approved (N={len(quadrant_data['b_only'])})",
    }

    centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    for qname, arr in quadrant_data.items():
        cfg = resolved[qname]
        color = cfg.get("color", "tab:blue")
        marker = cfg.get("marker", "o")
        markersize = cfg.get("markersize", None)
        label = cfg.get("label", default_labels[qname])
        per_quad_hist_kw = {**base_hist_kw, **cfg.get("hist_kwargs", {})}
        alpha = cfg.get("alpha", per_quad_hist_kw.get("alpha", 0.7 if scatter else 0.6))
        per_quad_hist_kw["alpha"] = alpha

        if scatter:
            if len(arr) == 0:
                top_ax.scatter(
                    [],
                    [],
                    color=color,
                    marker=marker,
                    s=markersize,
                    edgecolors="k",
                    alpha=alpha,
                    label=label,
                    zorder=3,
                )
                continue

            weights = per_quad_hist_kw.get("weights", None)
            heights, _ = np.histogram(arr, bins=bin_edges, density=density, weights=weights)
            top_ax.scatter(
                centres,
                heights,
                color=color,
                marker=marker,
                s=markersize,
                edgecolors="k",
                alpha=alpha,
                label=label,
                zorder=3,
            )
        else:
            if len(arr) == 0:
                # Add a legend proxy even for empty quadrants
                top_ax.hist([], bins=bin_edges, color=color, label=label, **per_quad_hist_kw)
                continue

            top_ax.hist(
                arr,
                bins=bin_edges,
                color=color,
                density=density,
                label=label,
                **per_quad_hist_kw,
            )

    # ------------------------------------------------------------------
    # Plot single-model approval percentages on bottom_ax
    # ------------------------------------------------------------------
    counts_total, _ = np.histogram(all_values, bins=bin_edges)
    valid_bins = counts_total > 0

    for qname in ["a_only", "b_only"]:
        cfg = resolved[qname]
        color = cfg.get("color", "tab:blue")
        marker = cfg.get("marker", "o")
        markersize = cfg.get("markersize", None)
        per_quad_hist_kw = {**base_hist_kw, **cfg.get("hist_kwargs", {})}
        alpha = cfg.get("alpha", per_quad_hist_kw.get("alpha", 0.7 if scatter else 0.6))
        per_quad_hist_kw["alpha"] = alpha
        arr = quadrant_data[qname]

        if len(arr) == 0 or not np.any(valid_bins):
            pct = np.zeros_like(centres)
        else:
            counts_q, _ = np.histogram(arr, bins=bin_edges)
            pct = np.divide(
                counts_q * 100.0,
                counts_total,
                out=np.zeros_like(counts_q, dtype=float),
                where=valid_bins,
            )

        if scatter:
            if len(arr) == 0 or not np.any(valid_bins):
                bottom_ax.scatter(
                    [],
                    [],
                    color=color,
                    marker=marker,
                    s=markersize,
                    edgecolors="k",
                    alpha=alpha,
                    zorder=3,
                )
            else:
                bottom_ax.scatter(
                    centres[valid_bins],
                    pct[valid_bins],
                    color=color,
                    marker=marker,
                    s=markersize,
                    edgecolors="k",
                    alpha=alpha,
                    zorder=3,
                )
        else:
            if len(arr) == 0:
                bottom_ax.hist([], bins=bin_edges, color=color, **per_quad_hist_kw)
            else:
                bottom_ax.hist(
                    centres,
                    bins=bin_edges,
                    weights=pct,
                    color=color,
                    **per_quad_hist_kw,
                )

    # ------------------------------------------------------------------
    # Axes formatting
    # ------------------------------------------------------------------
    top_ax.set_ylabel("Density" if density else "Counts", fontsize="small")
    top_ax.grid(linestyle="--", alpha=0.2, color="k")
    top_ax.tick_params(axis="x", which="both", labelbottom=False)
    top_ax.tick_params(axis="y", which="both", labelsize="small")

    if title is not None:
        top_ax.set_title(title, fontsize="large")

    legend_kw: dict = {"loc": "best", "fontsize": "small"}
    if legend_kwargs is not None:
        legend_kw.update(legend_kwargs)
    top_ax.legend(**legend_kw)

    bottom_ax.set_xlabel(var_label, fontsize="small", loc='left')
    bottom_ax.set_ylabel("Disagreement (%)", fontsize="small")
    bottom_ax.grid(linestyle="--", alpha=0.2, color="k")
    bottom_ax.tick_params(axis="x", which="both", labelsize="small")
    bottom_ax.tick_params(axis="y", which="both", labelsize="small")

    return fig, top_ax, bottom_ax, quadrant_data

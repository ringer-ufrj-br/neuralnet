import pytest
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from neuralnet.plotting import (
    plot_metrics_heatmap,
    plot_aggregated_metrics_grid,
    plot_joint_hist2d_with_marginals,
    plot_error_distributions_by_bits,
    quadrant_plot,
)


@pytest.fixture
def sample_results_df() -> pl.DataFrame:
    records = []
    for ib in [1, 2, 3]:
        for fb in [4, 6, 8, 10]:
            total = ib + fb
            mse = 10.0 ** (-(total / 3.0))
            mae = 10.0 ** (-(total / 3.5))
            mape = 100.0 * 10.0 ** (-(total / 4.0))
            max_err = 10.0 ** (-(total / 2.5))
            kl = 10.0 ** (-(total / 2.0))
            hist_kl = 10.0 ** (-(total / 2.2))
            records.append(
                {
                    "integer_bits": ib,
                    "fractional_bits": fb,
                    "total_bits": total,
                    "mse": mse,
                    "rmse": np.sqrt(mse),
                    "mae": mae,
                    "mape": mape,
                    "max_error": max_err,
                    "kl_divergence": kl,
                    "hist_kl_divergence": hist_kl,
                }
            )
    return pl.DataFrame(records)


def test_plot_metrics_heatmap(sample_results_df: pl.DataFrame):
    fig, ax = plot_metrics_heatmap(
        data=sample_results_df,
        metric_col="mse",
        x_col="fractional_bits",
        y_col="integer_bits",
        title="MSE Heatmap",
        annot=True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "MSE Heatmap"
    assert ax.get_xlabel() == "Fractional Bits ($fb$)"
    assert ax.get_ylabel() == "Integer Bits ($ib$)"
    plt.close(fig)


def test_plot_aggregated_metrics_grid(sample_results_df: pl.DataFrame):
    metrics = ["mse", "mae", "mape", "max_error", "kl_divergence", "hist_kl_divergence"]
    fig, axes = plot_aggregated_metrics_grid(
        data=sample_results_df,
        metrics=metrics,
        ncols=3,
        log_scale=True,
        annot=True,
    )
    assert isinstance(fig, plt.Figure)
    assert axes.shape == (2, 3)
    plt.close(fig)


def test_plot_joint_hist2d_with_marginals():
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.uniform(0.01, 1.0, size=n)
    y = x * 0.05 + rng.normal(0.0, 0.01, size=n)

    fig, main_ax, top_ax, right_ax = plot_joint_hist2d_with_marginals(
        x=x,
        y=y,
        xlabel="True Normalized Ring Value",
        ylabel="Quantization Error",
        title="Joint Error Distribution",
        bins=30,
        log_counts=True,
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(main_ax, plt.Axes)
    assert isinstance(top_ax, plt.Axes)
    assert isinstance(right_ax, plt.Axes)
    assert main_ax.get_xlabel() == "True Normalized Ring Value"
    assert main_ax.get_ylabel() == "Quantization Error"
    plt.close(fig)


def test_plot_joint_hist2d_with_marginals_polars_df():
    rng = np.random.default_rng(42)
    n = 500
    df = pl.DataFrame(
        {
            "fractional_bits": rng.choice([4, 6, 8, 10], size=n),
            "sample_mae": rng.exponential(scale=0.01, size=n) + 1e-6,
        }
    )

    fig, main_ax, top_ax, right_ax = plot_joint_hist2d_with_marginals(
        data=df,
        x_col="fractional_bits",
        y_col="sample_mae",
        xlabel="Fractional Bits",
        ylabel="Sample MAE",
        yscale="log",
        bins=20,
    )

    assert isinstance(fig, plt.Figure)
    assert main_ax.get_yscale() == "log"
    plt.close(fig)


def test_plot_error_distributions_by_bits():
    rng = np.random.default_rng(42)
    n = 200

    diff_dfs = {}
    for ib in [1, 2]:
        for fb in [4, 6, 8]:
            df = pl.DataFrame(
                {
                    "sample_mae": rng.exponential(scale=1.0 / (ib + fb), size=n) + 1e-6,
                }
            )
            diff_dfs[(ib, fb)] = df

    fig, ax = plot_error_distributions_by_bits(
        diff_dfs=diff_dfs,
        metric_col="sample_mae",
        fixed_bit_type="integer_bits",
        fixed_bit_val=1,
        xlabel="Sample MAE",
        bins=30,
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fixtures for quadrant_plot tests
# ---------------------------------------------------------------------------

@pytest.fixture
def quadrant_df() -> pl.DataFrame:
    """Balanced synthetic dataset with known quadrant memberships."""
    rng = np.random.default_rng(0)
    n = 200
    # Build explicit boolean arrays so we know exactly how many fall in each quadrant.
    # 50 each: both_false, both_true, a_only, b_only
    model_a = np.array([False] * 50 + [True] * 50 + [True] * 50 + [False] * 50)
    model_b = np.array([False] * 50 + [True] * 50 + [False] * 50 + [True] * 50)
    variable = rng.normal(0.0, 1.0, size=n)
    return pl.DataFrame(
        {
            "modelA": model_a,
            "modelB": model_b,
            "score": variable,
        }
    )


# ---------------------------------------------------------------------------
# quadrant_plot tests
# ---------------------------------------------------------------------------

class TestQuadrantPlot:
    """Tests for :func:`neuralnet.plotting.quadrant_plot`."""

    def test_returns_axes_and_quadrant_data(self, quadrant_df):
        """Function should return the used Axes and a dict with four arrays."""
        fig, ax = plt.subplots()
        result_ax, qdata = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA", "label": "Model A"},
            model_b_col={"col": "modelB", "label": "Model B"},
            variable_col={"col": "score", "label": "Score"},
            ax=ax,
        )
        assert result_ax is ax
        assert isinstance(result_ax, plt.Axes)
        assert set(qdata.keys()) == {"both_false", "both_true", "a_only", "b_only"}
        plt.close(fig)

    def test_quadrant_sample_counts(self, quadrant_df):
        """Each quadrant array must contain exactly the right samples."""
        _, qdata = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
        )
        assert len(qdata["both_false"]) == 50
        assert len(qdata["both_true"]) == 50
        assert len(qdata["a_only"]) == 50
        assert len(qdata["b_only"]) == 50
        plt.close("all")

    def test_axes_labels(self, quadrant_df):
        """x-label should reflect variable_col label; y-label should be 'Density'."""
        _, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score", "label": "My Score"},
        )
        ax = plt.gca()
        assert ax.get_xlabel() == "My Score"
        assert ax.get_ylabel() == "Density"
        plt.close("all")

    def test_default_xlabel_is_column_name(self, quadrant_df):
        """When label is omitted from variable_col the column name should be used."""
        _, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
        )
        ax = plt.gca()
        assert ax.get_xlabel() == "score"
        plt.close("all")

        _, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col="score",
        )
        ax = plt.gca()
        assert ax.get_xlabel() == "score"
        plt.close("all")

    def test_title_is_set(self, quadrant_df):
        """When title is provided it should appear on the axes."""
        fig, ax = plt.subplots()
        quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score", "label": "Score"},
            title="Test Title",
            ax=ax,
        )
        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_density_false_ylabel(self, quadrant_df):
        """With density=False the y-label should be 'Counts'."""
        fig, ax = plt.subplots()
        quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            density=False,
            ax=ax,
        )
        assert ax.get_ylabel() == "Counts"
        plt.close(fig)

    def test_custom_quadrant_config_color(self, quadrant_df):
        """Custom color in quadrant_config should be reflected in the plotted artist."""
        fig, ax = plt.subplots()
        quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            quadrant_config={"both_true": {"color": "crimson", "label": "Both accept"}},
            ax=ax,
        )
        # The legend texts should contain our custom label
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("Both accept" in t for t in legend_texts)
        plt.close(fig)

    def test_scatter_mode_only_plots_markers(self, quadrant_df):
        """With scatter=True axes should contain only markers."""
        from matplotlib.collections import PathCollection

        fig, ax = plt.subplots()
        quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA", "label": "Model A"},
            model_b_col={"col": "modelB", "label": "Model B"},
            variable_col={"col": "score", "label": "Score"},
            scatter=True,
            ax=ax,
        )
        scatter_artists = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatter_artists) > 0
        # No histogram bins/lines/patches drawn
        assert len(ax.patches) == 0
        assert len(ax.lines) == 0
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 4
        plt.close(fig)

    def test_scatter_false_plots_histograms_only(self, quadrant_df):
        """With scatter=False the axes should contain histogram step artists and no PathCollection."""
        from matplotlib.collections import PathCollection

        fig, ax = plt.subplots()
        quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            scatter=False,
            ax=ax,
        )
        scatter_artists = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatter_artists) == 0
        assert len(ax.lines) > 0 or len(ax.patches) > 0
        plt.close(fig)

    def test_invalid_scatter_type_raises(self, quadrant_df):
        """Passing a non-bool to scatter should raise TypeError."""
        for invalid in [{"scatter": True}, "True", 1, [True]]:
            with pytest.raises(TypeError, match="scatter must be a bool"):
                quadrant_plot(
                    data=quadrant_df,
                    model_a_col={"col": "modelA"},
                    model_b_col={"col": "modelB"},
                    variable_col={"col": "score"},
                    scatter=invalid,
                )
            plt.close("all")

    def test_lazy_frame_input(self, quadrant_df):
        """Should accept a LazyFrame and produce identical quadrant data."""
        _, qdata_eager = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
        )
        plt.close("all")

        _, qdata_lazy = quadrant_plot(
            data=quadrant_df.lazy(),
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
        )
        plt.close("all")

        for key in qdata_eager:
            np.testing.assert_array_equal(
                np.sort(qdata_eager[key]), np.sort(qdata_lazy[key])
            )

    def test_non_boolean_column_raises(self, quadrant_df):
        """Passing a non-Boolean prediction column should raise TypeError."""
        df_bad = quadrant_df.with_columns(pl.col("modelA").cast(pl.Int32))
        with pytest.raises(TypeError, match="Boolean"):
            quadrant_plot(
                data=df_bad,
                model_a_col={"col": "modelA"},
                model_b_col={"col": "modelB"},
                variable_col={"col": "score"},
            )
        plt.close("all")

    def test_explicit_bin_edges(self, quadrant_df):
        """Passing a np.ndarray for bins should be accepted without error."""
        edges = np.linspace(-4.0, 4.0, 21)
        fig, ax = plt.subplots()
        result_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            bins=edges,
            ax=ax,
        )
        assert isinstance(result_ax, plt.Axes)
        plt.close(fig)

    def test_uses_current_ax_when_none_passed(self, quadrant_df):
        """When ax=None the function should draw into plt.gca()."""
        fig, expected_ax = plt.subplots()
        plt.sca(expected_ax)
        result_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
        )
        assert result_ax is expected_ax
        plt.close(fig)

    def test_legend_contains_all_four_quadrants(self, quadrant_df):
        """The legend should have exactly four entries (one per quadrant)."""
        fig, ax = plt.subplots()
        quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            ax=ax,
        )
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 4
        plt.close(fig)

    def test_custom_hist_kwargs(self, quadrant_df):
        """Custom hist_kwargs should be forwarded without error."""
        fig, ax = plt.subplots()
        quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            hist_kwargs={"linewidth": 3.0, "alpha": 0.9},
            ax=ax,
        )
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_empty_quadrant_scatter_and_hist(self):
        """Empty quadrants should not crash in hist, scatter, or plot modes."""
        df = pl.DataFrame(
            {
                "modelA": [True, True, True],
                "modelB": [True, True, True],
                "score": [1.0, 2.0, 3.0],
            }
        )
        for scatter_val in [True, False]:
            fig, ax = plt.subplots()
            quadrant_plot(
                data=df,
                model_a_col={"col": "modelA"},
                model_b_col={"col": "modelB"},
                variable_col={"col": "score", "label": "Score"},
                scatter=scatter_val,
                ax=ax,
            )
            assert len(ax.get_legend().get_texts()) == 4
            plt.close(fig)

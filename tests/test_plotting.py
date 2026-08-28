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

    def test_returns_figure_axes_and_quadrant_data(self, quadrant_df):
        """Function should return the Figure, two Axes, and a dict with four arrays."""
        result_fig, top_ax, bottom_ax, qdata = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA", "label": "Model A"},
            model_b_col={"col": "modelB", "label": "Model B"},
            variable_col={"col": "score", "label": "Score"},
        )
        assert isinstance(result_fig, plt.Figure)
        assert isinstance(top_ax, plt.Axes)
        assert isinstance(bottom_ax, plt.Axes)
        assert set(qdata.keys()) == {"both_false", "both_true", "a_only", "b_only"}
        # Both axes share x-axis
        assert bottom_ax._sharex is top_ax or top_ax._sharex is bottom_ax
        plt.close(result_fig)

    def test_custom_fig_kwargs(self, quadrant_df):
        """Passing fig_kwargs should configure the created Figure."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            fig_kwargs={"figsize": (10, 8)},
        )
        assert isinstance(fig, plt.Figure)
        np.testing.assert_allclose(fig.get_size_inches(), (10, 8))
        plt.close(fig)

    def test_quadrant_sample_counts(self, quadrant_df):
        """Each quadrant array must contain exactly the right samples."""
        _, _, _, qdata = quadrant_plot(
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
        """x-label on bottom_ax should reflect variable_col; y-label on top_ax should be 'Density'."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score", "label": "My Score"},
        )
        assert bottom_ax.get_xlabel() == "My Score"
        assert top_ax.get_ylabel() == "Density"
        assert bottom_ax.get_ylabel() == "Disagreement (%)"
        plt.close(fig)

    def test_default_xlabel_is_column_name(self, quadrant_df):
        """When label is omitted from variable_col the column name should be used."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
        )
        assert bottom_ax.get_xlabel() == "score"
        plt.close(fig)

        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col="score",
        )
        assert bottom_ax.get_xlabel() == "score"
        plt.close(fig)

    def test_title_is_set_on_top_ax(self, quadrant_df):
        """When title is provided it should appear on the top axes."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score", "label": "Score"},
            title="Test Title",
        )
        assert top_ax.get_title() == "Test Title"
        assert bottom_ax.get_title() == ""
        plt.close(fig)

    def test_density_false_ylabel(self, quadrant_df):
        """With density=False the top y-label should be 'Counts'."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            density=False,
        )
        assert top_ax.get_ylabel() == "Counts"
        assert bottom_ax.get_ylabel() == "Disagreement (%)"
        plt.close(fig)

    def test_custom_quadrant_config_color(self, quadrant_df):
        """Custom color in quadrant_config should be reflected in the plotted artist."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            quadrant_config={"both_true": {"color": "crimson", "label": "Both accept"}},
        )
        # The legend texts on top_ax should contain our custom label
        legend_texts = [t.get_text() for t in top_ax.get_legend().get_texts()]
        assert any("Both accept" in t for t in legend_texts)
        plt.close(fig)

    def test_custom_quadrant_config_markersize(self, quadrant_df):
        """Custom markersize in quadrant_config should be reflected in scatter collections."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            quadrant_config={
                "both_true": {"markersize": 80.0},
                "a_only": {"markersize": 50.0},
            },
            scatter=True,
        )
        top_collections = top_ax.collections
        assert len(top_collections) == 4
        np.testing.assert_array_equal(top_collections[1].get_sizes(), [80.0])
        np.testing.assert_array_equal(top_collections[2].get_sizes(), [50.0])

        bottom_collections = bottom_ax.collections
        assert len(bottom_collections) == 2
        np.testing.assert_array_equal(bottom_collections[0].get_sizes(), [50.0])
        plt.close(fig)

    def test_scatter_mode_only_plots_markers(self, quadrant_df):
        """With scatter=True axes should contain markers and no step histogram lines/patches."""
        from matplotlib.collections import PathCollection

        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA", "label": "Model A"},
            model_b_col={"col": "modelB", "label": "Model B"},
            variable_col={"col": "score", "label": "Score"},
            scatter=True,
        )
        top_scatter_artists = [c for c in top_ax.collections if isinstance(c, PathCollection)]
        bottom_scatter_artists = [c for c in bottom_ax.collections if isinstance(c, PathCollection)]
        assert len(top_scatter_artists) > 0
        assert len(bottom_scatter_artists) == 2  # a_only and b_only
        # No histogram bins/lines/patches drawn
        assert len(top_ax.patches) == 0
        assert len(top_ax.lines) == 0
        assert len(bottom_ax.patches) == 0
        assert len(bottom_ax.lines) == 0
        legend = top_ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 4
        assert bottom_ax.get_legend() is None
        plt.close(fig)

    def test_scatter_false_plots_histograms_only(self, quadrant_df):
        """With scatter=False both axes should contain histogram step artists and no PathCollection."""
        from matplotlib.collections import PathCollection

        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            scatter=False,
        )
        top_scatter = [c for c in top_ax.collections if isinstance(c, PathCollection)]
        bottom_scatter = [c for c in bottom_ax.collections if isinstance(c, PathCollection)]
        assert len(top_scatter) == 0
        assert len(bottom_scatter) == 0
        assert len(top_ax.lines) > 0 or len(top_ax.patches) > 0
        assert len(bottom_ax.lines) > 0 or len(bottom_ax.patches) > 0
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
        _, _, _, qdata_eager = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
        )
        plt.close("all")

        _, _, _, qdata_lazy = quadrant_plot(
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
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            bins=edges,
        )
        assert isinstance(top_ax, plt.Axes)
        assert isinstance(bottom_ax, plt.Axes)
        plt.close(fig)

    def test_creates_fig_when_none_passed(self, quadrant_df):
        """When fig_kwargs=None the function should create a new Figure."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_invalid_fig_kwargs_raises(self, quadrant_df):
        """Passing a non-dict to fig_kwargs should raise TypeError."""
        with pytest.raises(TypeError, match="fig_kwargs must be a dict or None"):
            quadrant_plot(
                data=quadrant_df,
                model_a_col={"col": "modelA"},
                model_b_col={"col": "modelB"},
                variable_col={"col": "score"},
                fig_kwargs="invalid",
            )
        plt.close("all")

    def test_legend_contains_all_four_quadrants_on_top_only(self, quadrant_df):
        """The legend should have exactly four entries on top_ax and none on bottom_ax."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
        )
        legend = top_ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 4
        assert bottom_ax.get_legend() is None
        plt.close(fig)

    def test_custom_hist_kwargs(self, quadrant_df):
        """Custom hist_kwargs should be forwarded without error."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            hist_kwargs={"linewidth": 3.0, "alpha": 0.9},
        )
        assert isinstance(top_ax, plt.Axes)
        assert isinstance(bottom_ax, plt.Axes)
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
            fig, top_ax, bottom_ax, _ = quadrant_plot(
                data=df,
                model_a_col={"col": "modelA"},
                model_b_col={"col": "modelB"},
                variable_col={"col": "score", "label": "Score"},
                scatter=scatter_val,
            )
            assert len(top_ax.get_legend().get_texts()) == 4
            assert bottom_ax.get_legend() is None
            plt.close(fig)

    def test_single_model_approval_percentage_values(self):
        """Percentages plotted on bottom_ax must equal (N_only / N_total) * 100% per bin."""
        # 1 bin scenario: [0.0, 10.0]
        # Total: 10 samples (2 both_false, 3 both_true, 4 a_only, 1 b_only)
        # a_only percentage: 4/10 = 40.0%
        # b_only percentage: 1/10 = 10.0%
        df = pl.DataFrame(
            {
                "modelA": [False, False, True, True, True, True, True, True, True, False],
                "modelB": [False, False, True, True, True, False, False, False, False, True],
                "score": [5.0] * 10,
            }
        )
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            bins=np.array([0.0, 10.0]),
            scatter=True,
        )
        # bottom_ax collections: collection 0 is a_only, collection 1 is b_only
        collections = bottom_ax.collections
        assert len(collections) == 2
        offsets_a = collections[0].get_offsets()
        offsets_b = collections[1].get_offsets()
        np.testing.assert_allclose(offsets_a[0, 1], 40.0)
        np.testing.assert_allclose(offsets_b[0, 1], 10.0)
        plt.close(fig)

    def test_maintains_colors_and_markers_on_bottom_ax(self):
        """bottom_ax scatter must preserve custom colors, markers, and markersize configured for quadrants."""
        df = pl.DataFrame(
            {
                "modelA": [True, False, True, False],
                "modelB": [False, True, True, False],
                "score": [1.0, 2.0, 3.0, 4.0],
            }
        )
        custom_config = {
            "a_only": {"color": "magenta", "marker": "D", "markersize": 45.0},
            "b_only": {"color": "cyan", "marker": "P", "markersize": 60.0},
        }
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
            quadrant_config=custom_config,
            scatter=True,
        )
        assert len(bottom_ax.collections) == 2
        np.testing.assert_array_equal(bottom_ax.collections[0].get_sizes(), [45.0])
        np.testing.assert_array_equal(bottom_ax.collections[1].get_sizes(), [60.0])
        plt.close(fig)

    def test_axes_share_x(self, quadrant_df):
        """Setting xlim on top_ax should automatically update bottom_ax xlim."""
        fig, top_ax, bottom_ax, _ = quadrant_plot(
            data=quadrant_df,
            model_a_col={"col": "modelA"},
            model_b_col={"col": "modelB"},
            variable_col={"col": "score"},
        )
        top_ax.set_xlim(-2.5, 2.5)
        np.testing.assert_allclose(bottom_ax.get_xlim(), (-2.5, 2.5))
        plt.close(fig)



import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from neuralnet.metrics import (
    enhanced_confusion_matrix,
    enhanced_confusion_matrix_from_polars,
    polars_sp_index,
    sp_index,
)


# ---------------------------------------------------------------------------
# sp_index
# ---------------------------------------------------------------------------


class TestSpIndex:
    def test_perfect_classifier(self):
        """TPR=1, FPR=0 → SP = 1."""
        assert sp_index(1.0, 0.0) == pytest.approx(1.0)

    def test_random_classifier(self):
        """TPR=0.5, FPR=0.5 → SP = 0.5."""
        assert sp_index(0.5, 0.5) == pytest.approx(0.5)

    def test_worst_classifier(self):
        """TPR=0, FPR=1 → SP = 0."""
        assert sp_index(0.0, 1.0) == pytest.approx(0.0)

    def test_typical_values(self):
        tpr, fpr = 0.9, 0.1
        expected = math.sqrt(math.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))
        assert sp_index(tpr, fpr) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# polars_sp_index
# ---------------------------------------------------------------------------


class TestPolarsSpIndex:
    def test_matches_scalar_sp_index(self):
        tpr_vals = [1.0, 0.5, 0.0, 0.9]
        fpr_vals = [0.0, 0.5, 1.0, 0.1]

        lf = pl.LazyFrame({"tpr": tpr_vals, "fpr": fpr_vals})
        result = (
            lf.select(polars_sp_index(pl.col("tpr"), pl.col("fpr")).alias("sp"))
            .collect()["sp"]
            .to_list()
        )
        expected = [sp_index(t, f) for t, f in zip(tpr_vals, fpr_vals)]

        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# enhanced_confusion_matrix
# ---------------------------------------------------------------------------


class TestEnhancedConfusionMatrix:
    def _make_cm(self):
        import numpy as np

        tn = np.array([0, 5, 10])
        tp = np.array([10, 8, 5])
        fn = np.array([0, 2, 5])
        fp = np.array([10, 5, 0])
        thresholds = np.array([0.3, 0.5, 0.7])
        return enhanced_confusion_matrix(tn, tp, fn, fp, thresholds)

    def test_keys_present(self):
        cm = self._make_cm()
        for key in ("tn", "tp", "fn", "fp", "thresholds", "positives", "negatives",
                    "total", "correct", "incorrect", "accuracy", "tpr", "fpr", "sp", "auc"):
            assert key in cm, f"Missing key: {key}"

    def test_totals(self):
        cm = self._make_cm()
        assert cm["positives"] + cm["negatives"] == cm["total"]

    def test_tpr_range(self):
        import numpy as np
        cm = self._make_cm()
        assert all(0.0 <= v <= 1.0 for v in np.asarray(cm["tpr"]))

    def test_fpr_range(self):
        import numpy as np
        cm = self._make_cm()
        assert all(0.0 <= v <= 1.0 for v in np.asarray(cm["fpr"]))

    def test_sp_range(self):
        import numpy as np
        cm = self._make_cm()
        assert all(0.0 <= v <= 1.0 for v in np.asarray(cm["sp"]))

    def test_accuracy_range(self):
        import numpy as np
        cm = self._make_cm()
        assert all(0.0 <= v <= 1.0 for v in np.asarray(cm["accuracy"]))

    def test_auc_is_scalar(self):
        cm = self._make_cm()
        assert isinstance(cm["auc"], float)


# ---------------------------------------------------------------------------
# enhanced_confusion_matrix_from_polars
# ---------------------------------------------------------------------------


class TestEnhancedConfusionMatrixFromPolars:
    @pytest.fixture
    def binary_data(self):
        return {
            "label": [1, 1, 0, 0, 1, 0],
            "score": [0.9, 0.8, 0.4, 0.3, 0.6, 0.7],
        }

    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_returns_same_type(self, frame_factory, binary_data):
        data = frame_factory(binary_data)
        result = enhanced_confusion_matrix_from_polars(data, "label", "score")
        assert isinstance(result, type(data))

    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_output_columns(self, frame_factory, binary_data):
        data = frame_factory(binary_data)
        result = enhanced_confusion_matrix_from_polars(data, "label", "score")
        if isinstance(result, pl.LazyFrame):
            result = result.collect()
        expected_cols = {"threshold", "tp", "fp", "fn", "tn", "positives",
                         "negatives", "total", "correct", "incorrect",
                         "tpr", "fpr", "accuracy", "sp"}
        assert expected_cols.issubset(set(result.columns))

    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_tp_plus_fn_equals_positives(self, frame_factory, binary_data):
        data = frame_factory(binary_data)
        result = enhanced_confusion_matrix_from_polars(data, "label", "score")
        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        for row in result.iter_rows(named=True):
            assert row["tp"] + row["fn"] == row["positives"]

    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_fp_plus_tn_equals_negatives(self, frame_factory, binary_data):
        data = frame_factory(binary_data)
        result = enhanced_confusion_matrix_from_polars(data, "label", "score")
        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        for row in result.iter_rows(named=True):
            assert row["fp"] + row["tn"] == row["negatives"]

    @pytest.mark.parametrize("frame_factory", [pl.DataFrame, pl.LazyFrame])
    def test_sorted_descending_by_threshold(self, frame_factory, binary_data):
        data = frame_factory(binary_data)
        result = enhanced_confusion_matrix_from_polars(data, "label", "score")
        if isinstance(result, pl.LazyFrame):
            result = result.collect()

        thresholds = result["threshold"].to_list()
        assert thresholds == sorted(thresholds, reverse=True)

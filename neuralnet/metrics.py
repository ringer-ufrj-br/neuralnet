from typing import TypedDict, TYPE_CHECKING, overload
import numpy as np
import numpy.typing as npt
from numbers import Real
import polars as pl

if TYPE_CHECKING:
    import torch

from .numpy import Numpy1DIntegerArray, Numpy1DFloatArray


def sp_index(tpr: Real, fpr: Real) -> Real:
    """Calculate the SP index"""
    return np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))


def torch_sp_index(tpr: "torch.Tensor", fpr: "torch.Tensor") -> "torch.Tensor":
    """Calculate the SP index for PyTorch tensors"""
    return torch.sqrt(torch.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))


def polars_sp_index(tpr: pl.Expr, fpr: pl.Expr) -> pl.Expr:
    """Calculate the SP index for Polars expressions"""
    return ((tpr * (1 - fpr)).sqrt() * (0.5 * (tpr + (1 - fpr)))).sqrt()


class MaxSPDict(TypedDict):
    threshold: float
    tn: int
    tp: int
    fn: int
    fp: int
    acc: float
    tpr: float
    fpr: float
    sp: float


class EnhancedConfusionMatrixDict(TypedDict):
    tn: Numpy1DIntegerArray
    tp: Numpy1DIntegerArray
    fn: Numpy1DIntegerArray
    fp: Numpy1DIntegerArray
    thresholds: Numpy1DFloatArray
    total: np.integer
    correct: np.integer
    incorrect: np.integer
    positives: np.integer
    negatives: np.integer
    accuracy: Numpy1DFloatArray
    tpr: Numpy1DFloatArray
    fpr: Numpy1DFloatArray
    sp: Numpy1DFloatArray
    auc: np.floating


def enhanced_confusion_matrix_from_preds(
    y_true: Numpy1DFloatArray, y_pred: Numpy1DFloatArray
) -> EnhancedConfusionMatrixDict:
    from sklearn.metrics import confusion_matrix_at_thresholds

    cm: tuple[
        Numpy1DIntegerArray,
        Numpy1DIntegerArray,
        Numpy1DIntegerArray,
        Numpy1DIntegerArray,
        Numpy1DFloatArray,
    ] = confusion_matrix_at_thresholds(y_true, y_pred, pos_label=1)
    tn, fp, fn, tp, thresholds = cm
    return enhanced_confusion_matrix(tn, tp, fn, fp, thresholds)


def enhanced_confusion_matrix(
    tn: npt.NDArray[np.integer | np.floating],
    tp: npt.NDArray[np.integer | np.floating],
    fn: npt.NDArray[np.integer | np.floating],
    fp: npt.NDArray[np.integer | np.floating],
    thresholds: npt.NDArray[np.floating],
) -> EnhancedConfusionMatrixDict:
    enhanced_cm = {
        "tn": tn.tolist(),
        "tp": tp.tolist(),
        "fn": fn.tolist(),
        "fp": fp.tolist(),
        "thresholds": thresholds.tolist(),
    }
    argsort = np.argsort(thresholds)
    thresholds = thresholds[argsort]
    tn = tn[argsort].astype(int)
    tp = tp[argsort].astype(int)
    fn = fn[argsort].astype(int)
    fp = fp[argsort].astype(int)
    enhanced_cm["positives"] = tp[0] + fn[0]
    enhanced_cm["negatives"] = tn[0] + fp[0]
    enhanced_cm["total"] = enhanced_cm["positives"] + enhanced_cm["negatives"]
    enhanced_cm["correct"] = tn + tp
    enhanced_cm["incorrect"] = fn + fp
    acc = (tn + tp) / (tn + tp + fn + fp)
    enhanced_cm["accuracy"] = acc
    tpr = (tp) / (tp + fn)
    enhanced_cm["tpr"] = tpr
    fpr = (fp) / (fp + tn)
    enhanced_cm["fpr"] = fpr
    sp = sp_index(tpr, fpr)
    enhanced_cm["sp"] = sp
    tpr_argsort = np.argsort(tpr)

    enhanced_cm["auc"] = float(np.trapezoid(fpr[tpr_argsort], tpr[tpr_argsort]))

    return enhanced_cm


@overload
def enhanced_confusion_matrix_from_polars(data: pl.DataFrame, label_col: str, score_col: str) -> pl.DataFrame: ...


@overload
def enhanced_confusion_matrix_from_polars(data: pl.LazyFrame, label_col: str, score_col: str) -> pl.LazyFrame: ...


def enhanced_confusion_matrix_from_polars(data: pl.LazyFrame | pl.DataFrame, label_col: str, score_col: str):
    """
    Computes confusion matrix components (TN, FP, FN, TP).
    """

    # Cast target to integer to handle booleans or floats seamlessly
    label_expr = pl.col(label_col).cast(pl.Int32)

    cm_data = (
        data.drop_nulls(subset=[label_col, score_col])
        # 2. Group by threshold to aggregate tied neural network scores
        .group_by(score_col)
        .agg(label_expr.sum().alias("pos_count"), (pl.len() - label_expr.sum()).alias("neg_count"))
        # 3. Sort thresholds descending (highest prediction confidence first)
        .sort(score_col, descending=True)
        # 4. Calculate True Positives and False Positives cumulatively
        .select(
            pl.col("pos_count").cum_sum().alias("tp"),
            pl.col("neg_count").cum_sum().alias("fp"),
            pl.col("pos_count").sum().alias("positives"),
            pl.col("neg_count").sum().alias("negatives"),
        )
        .with_columns(
            (pl.col("positives") - pl.col("tp")).alias("fn"),
            (pl.col("negatives") - pl.col("fp")).alias("tn"),
            (pl.col("positives") + pl.col("negatives")).alias("total"),
        )
        .with_columns(
            (pl.col("tp") + pl.col("tn")).alias("correct"),
            (pl.col("fn") + pl.col("fp")).alias("incorrect"),
            (pl.col("tp")).truediv(pl.col("positives")).alias("tpr"),
            (pl.col("fp")).truediv(pl.col("negatives")).alias("fpr"),
            (pl.col("tp") + pl.col("tn")).truediv(pl.col("total")).alias("accuracy"),
        )
        .with_columns(polars_sp_index(pl.col("tpr"), pl.col("fpr")).alias("sp"))
    )

    return cm_data

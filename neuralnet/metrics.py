from typing import TypedDict, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from numbers import Real
if TYPE_CHECKING:
    import torch

from .numpy import Numpy1DIntegerArray, Numpy1DFloatArray


def sp_index(tpr: Real, fpr: Real) -> Real:
    """Calculate the SP index"""
    return np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))

def torch_sp_index(tpr: "torch.Tensor", fpr: "torch.Tensor") -> "torch.Tensor":
    """Calculate the SP index for PyTorch tensors"""
    return torch.sqrt(torch.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))


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
    max_sp: MaxSPDict


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

    sp_argmax = np.argmax(sp)
    max_sp_dict = {
        "argmax": sp_argmax,
        "threshold": thresholds[sp_argmax],
        "tn": tn[sp_argmax],
        "tp": tp[sp_argmax],
        "fn": fn[sp_argmax],
        "fp": fp[sp_argmax],
        "acc": acc[sp_argmax],
        "tpr": tpr[sp_argmax],
        "fpr": fpr[sp_argmax],
        "sp": sp[sp_argmax],
    }
    enhanced_cm["max_sp"] = max_sp_dict
    return enhanced_cm



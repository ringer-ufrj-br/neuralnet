from typing import TypedDict
import numpy as np
import numpy.typing as npt
from numbers import Real


def sp_index(tpr: Real, fpr: Real) -> Real:
    """Calculate the SP index"""
    return np.sqrt(np.sqrt(tpr * (1 - fpr)) * (0.5 * (tpr + (1 - fpr))))


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
    tn: list[int]
    tp: list[int]
    fn: list[int]
    fp: list[int]
    thresholds: list[float]
    total: int
    correct: int
    incorrect: int
    positives: int
    negatives: int
    accuracy: list[float]
    tpr: list[float]
    fpr: list[float]
    sp: list[float]
    auc: float
    max_sp: MaxSPDict


def enhanced_confusion_matrix(
    tn: npt.NDArray[np.integer],
    tp: npt.NDArray[np.integer],
    fn: npt.NDArray[np.integer],
    fp: npt.NDArray[np.integer],
    thresholds: npt.NDArray[np.floating],
) -> EnhancedConfusionMatrixDict:
    enhanced_cm = {
        "tn": tn.tolist(),
        "tp": tp.tolist(),
        "fn": fn.tolist(),
        "fp": fp.tolist(),
        "thresholds": thresholds.tolist(),
    }
    enhanced_cm["total"] = int((tn + tp + fn + fp)[0])
    enhanced_cm["correct"] = int((tn + tp))
    enhanced_cm["incorrect"] = int((fn + fp))
    enhanced_cm["positives"] = int((tp + fn)[0])
    enhanced_cm["negatives"] = int((tn + fp)[0])
    acc = (tn + tp) / (tn + tp + fn + fp)
    enhanced_cm["accuracy"] = acc.tolist()
    tpr = (tp) / (tp + fn)
    enhanced_cm["tpr"] = tpr.tolist()
    fpr = (fp) / (fp + tn)
    enhanced_cm["fpr"] = fpr.tolist()
    sp = sp_index(tpr, fpr)
    enhanced_cm["sp"] = sp.tolist()
    enhanced_cm["auc"] = float(np.trapz(tpr, fpr))

    sp_argmax = np.argmax(sp)
    max_sp_dict = {
        "argmax": int(sp_argmax),
        "threshold": float(thresholds[sp_argmax]),
        "tn": int(tn[sp_argmax]),
        "tp": int(tp[sp_argmax]),
        "fn": int(fn[sp_argmax]),
        "fp": int(fp[sp_argmax]),
        "acc": float(acc[sp_argmax]),
        "tpr": float(tpr[sp_argmax]),
        "fpr": float(fpr[sp_argmax]),
        "sp": float(sp[sp_argmax]),
    }
    enhanced_cm["max_sp"] = max_sp_dict
    return enhanced_cm

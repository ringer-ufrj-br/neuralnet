from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from functools import cached_property
import json
from pathlib import Path


@dataclass(frozen=True)
class ConfusionMatrix:
    loss: np.floating
    tn: npt.NDArray[np.integer]
    tp: npt.NDArray[np.integer]
    fn: npt.NDArray[np.integer]
    fp: npt.NDArray[np.integer]
    thresholds: npt.NDArray[np.floating]

    @cached_property
    def total(self) -> npt.NDArray[np.integer]:
        return self.tn + self.tp + self.fn + self.fp

    @cached_property
    def correct(self) -> npt.NDArray[np.integer]:
        return self.tn + self.tp

    @cached_property
    def incorrect(self) -> npt.NDArray[np.integer]:
        return self.fn + self.fp

    @cached_property
    def positives(self) -> npt.NDArray[np.integer]:
        return self.tp + self.fn

    @cached_property
    def negatives(self) -> npt.NDArray[np.integer]:
        return self.tn + self.fp

    @cached_property
    def accuracy(self) -> npt.NDArray[np.floating]:
        return np.where(self.total > 0, self.correct / self.total, np.nan)

    @cached_property
    def recall(self) -> npt.NDArray[np.floating]:
        return np.where(self.positives > 0, self.tp / self.positives, np.nan)

    @property
    def pd(self) -> npt.NDArray[np.floating]:
        return self.recall
    
    @property
    def tpr(self) -> npt.NDArray[np.floating]:
        return self.recall

    @cached_property
    def fpr(self) -> npt.NDArray[np.floating]:
        return np.where(self.negatives > 0, self.fp / self.negatives, np.nan)

    @cached_property
    def fa(self) -> npt.NDArray[np.floating]:
        return self.fpr

    @cached_property
    def sp_index(self) -> npt.NDArray[np.floating]:
        """Calculate the SP index for each threshold."""
        return np.sqrt(
            np.sqrt(self.pd * (1 - self.fa)) * (0.5 * (self.pd + (1 - self.fa)))
        )

    @cached_property
    def auc(self) -> float:
        """Calculate the AUC using the trapezoidal rule."""
        # Sort by FPR
        sorted_indices = np.argsort(self.fpr)
        sorted_fpr = self.fpr[sorted_indices]
        sorted_pd = self.pd[sorted_indices]
        return np.trapezoid(sorted_pd, sorted_fpr)

    def to_dict(self, full: bool = False) -> dict[str, list[float | int] | float]:
        res = {
            "loss": float(self.loss),
            "tn": self.tn.tolist(),
            "tp": self.tp.tolist(),
            "fn": self.fn.tolist(),
            "fp": self.fp.tolist(),
            "thresholds": self.thresholds.tolist(),
        }
        if not full:
            return res

        res["accuracy"] = self.accuracy.tolist()
        res["tpr"] = self.tpr.tolist()
        res["fpr"] = self.fpr.tolist()
        res["sp_index"] = self.sp_index.tolist()
        res["auc"] = float(self.auc)
        return res

    @classmethod
    def from_json(cls, filepath: str | Path) -> "ConfusionMatrix":
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(
            tn=np.array(data["tn"], dtype=int),
            tp=np.array(data["tp"], dtype=int),
            fn=np.array(data["fn"], dtype=int),
            fp=np.array(data["fp"], dtype=int),
            thresholds=np.array(data["thresholds"], dtype=float),
        )

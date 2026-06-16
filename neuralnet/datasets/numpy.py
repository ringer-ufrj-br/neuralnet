import numpy as np
from typing import Protocol, runtime_checkable

type NumpyDatasetReturnTypes = (
    tuple[np.ndarray, np.ndarray] |
    tuple[list[np.ndarray], np.ndarray] |
    tuple[np.ndarray, list[np.ndarray]] |
    tuple[list[np.ndarray], list[np.ndarray]]
)

@runtime_checkable
class NumpyDataset(Protocol):
    def train_numpy(self) -> NumpyDatasetReturnTypes: ...

    def val_numpy(self) -> NumpyDatasetReturnTypes: ...

    def test_numpy(self) -> NumpyDatasetReturnTypes: ...

    def predict_numpy(self) -> NumpyDatasetReturnTypes: ...

from typing import Protocol, runtime_checkable


@runtime_checkable
class ClassWeightedDataset(Protocol):
    def class_weights(self) -> dict[int, float]: ...


@runtime_checkable
class ClassLabelsDataset(Protocol):
    CLASS_LABELS: dict[int, str]


@runtime_checkable
class KFoldDataset(Protocol):
    def n_folds(self) -> int: ...

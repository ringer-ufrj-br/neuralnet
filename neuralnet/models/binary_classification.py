from typing import Annotated, Literal, Callable, NotRequired
from functools import cached_property
from pydantic import Field, ConfigDict, PrivateAttr, BaseModel

import numpy as np
import numpy.typing as npt

from .keras_factories import (
    KerasSequentialModelFactory,
    StandardEvaluationDict,
    StandardFitDict,
)
from ..layers.dense import DenseFactory
from ..layers.quantum import (
    BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory,
    StronglyEntanglingQuantumLayerWithAnglerEmbeddingFactory,
    HardwareEfficientQuantumLayerWithAnglerEmbeddingFactory,
)
from ..losses.binary_loss import BinaryCrossEntropyLossFactory
from ..optimizers.adam import AdamFactory
from ..datasets.numpy import NumpyDataset


type LayerType = Annotated[
    (
        DenseFactory
        | BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory
        | StronglyEntanglingQuantumLayerWithAnglerEmbeddingFactory
        | HardwareEfficientQuantumLayerWithAnglerEmbeddingFactory
    ),
    Field(discriminator="object_type", description="Layer configuration field."),
]

type OptimizerType = Annotated[
    AdamFactory,
    Field(discriminator="object_type", description="Optimizer configuration field."),
]

type LossType = Annotated[
    BinaryCrossEntropyLossFactory,
    Field(discriminator="object_type", description="Loss configuration field."),
]

type UpperThresholdLimit = Annotated[
    float,
    Field(description="Upper threshold limit for the metric."),
]

type LowerThresholdLimit = Annotated[
    float,
    Field(description="Lower threshold limit for the metric."),
]

type NumThresholds = Annotated[
    int,
    Field(
        gt=0,
        description="Number of thresholds for the metric.",
    ),
]

type FromLogitsType = Annotated[
    bool,
    Field(
        False,
        description="Whether the model output to consider is logits. When enabled considers the threshold limits as probabilities between 0 and 1",
    ),
]

type ModelDatasetType = Annotated[
    NumpyDataset,
    Field(description="Validation dataset as a tuple of (X_val, y_val)"),
]


class ThresholdStrategy(BaseModel):
    name: Literal["max_sp"] = Field(
        "max_sp",
        description="Strategy to select the best threshold. Currently only supports 'max_sp' which selects the threshold with the highest SP index.",
    )
    num_thresholds: int = Field(
        gt=0,
        description="Number of thresholds to evaluate for the strategy.",
    )
    lower_threshold: float = Field(
        ge=0,
        lt=1,
        description="Lower threshold limit for the strategy.",
    )
    upper_threshold: float = Field(
        gt=0,
        le=1,
        description="Upper threshold limit for the strategy.",
    )

    _thresholds: npt.NDArray[np.floating] | None = PrivateAttr(default=None)

    def model_post_init(self, context):
        res = super().model_post_init(context)
        if self.lower_threshold >= self.upper_threshold:
            raise ValueError("Lower threshold must be lower than upper threshold")
        if self.from_logits:
            if self.lower_threshold < 0 or self.upper_threshold > 1:
                raise ValueError(
                    "Thresholds must be between 0 and 1 when from_logits is True"
                )

        metric_thresholds = np.linspace(
            start=self.lower_threshold,
            stop=self.upper_threshold,
            num=self.num_thresholds,
        )
        self._thresholds = metric_thresholds

        return res

    def transform_thresholds(
        self, func: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
    ):
        if self._thresholds is None:
            raise ValueError("Thresholds not initialized")
        self._thresholds = func(self._thresholds)

    @property
    def thresholds(self) -> npt.NDArray[np.floating]:
        if self._thresholds is None:
            raise ValueError("Thresholds not initialized")
        return self._thresholds

    @cached_property
    def thresholds_list(self) -> list[float]:
        return self.thresholds.tolist()


class KerasSequentialBinaryClassificationModelFactory(KerasSequentialModelFactory):
    """
    Model configuration and utilities for training, evaluating, saving, and loading a Keras Sequential model for binary classification tasks. The model supports custom layers, optimizers, and loss functions, as well as configurable thresholds for evaluation metrics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_type: Literal["binary_classification_model"] = Field(
        "binary_classification_model",
        description="Name of the model. Should be 'binary_classification_model' for this class.",
    )

    layers: list[LayerType] = Field(
        ..., min_items=1, description="List of sequential model layers."
    )

    loss: LossType
    optimizer: OptimizerType
    from_logits: FromLogitsType
    threshold_strategy: ThresholdStrategy
    balance_class_weights: bool = Field(
        True,
        description="Whether to balance class weights during training. If True, the class weights will be set inversely proportional to the class frequencies.",
    )

    # Training related fields
    patience: int = Field(
        25,
        description="Number of epochs with no improvement after which training will be stopped.",
    )

    def as_keras(self):
        from .keras.binary_classification import BinaryClassificationModel

        return BinaryClassificationModel.from_factory(self)


class FitDict(StandardFitDict):
    loss: list[float]
    accuracy: list[float]
    max_sp_val: list[float]
    max_sp_pd_val: list[float]
    max_sp_fa_val: list[float]
    max_sp_threshold_val: list[float]


class EvaluationDict(StandardEvaluationDict):
    loss: float
    thresholds: list[float]
    true_negatives: list[float]
    true_positives: list[float]
    false_negatives: list[float]
    false_positives: list[float]
    tpr: list[float]
    weighted_tpr: NotRequired[list[float]]
    fpr: list[float]
    sp: list[float]
    max_sp: float
    max_sp_tpr: float
    max_sp_fpr: float
    max_sp_idx: int
    max_sp_threshold: float
    class_weighted: NotRequired["EvaluationDict"]

    weighted_true_negatives: NotRequired[list[float]]
    weighted_true_positives: NotRequired[list[float]]
    weighted_false_negatives: NotRequired[list[float]]
    weighted_false_positives: NotRequired[list[float]]
    weighted_fpr: NotRequired[list[float]]
    weighted_sp: NotRequired[list[float]]
    weighted_max_sp: NotRequired[float]
    weighted_max_sp_tpr: NotRequired[float]
    weighted_max_sp_fpr: NotRequired[float]

from pydantic import BaseModel, Field, JsonValue
from typing import Annotated, Literal, TypedDict
from datetime import datetime

from ...layers.dense import DenseFactory
from ...layers.quantum import (
    BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory,
    StronglyEntanglingQuantumLayerWithAnglerEmbeddingFactory,
    HardwareEfficientQuantumLayerWithAnglerEmbeddingFactory,
)
from ...losses.binary_loss import BinaryCrossEntropyLossFactory
from ...optimizers.adam import AdamFactory

type EpochsType = Annotated[
    int,
    Field(5000, description="Number of epochs to train the model.", gt=0),
]

type VerboseType = Annotated[
    int,
    Field(1, ge=0, le=2, description="Verbosity of the training."),
]

type OptimizerType = Annotated[
    AdamFactory,
    Field(discriminator="object_type", description="Optimizer configuration field."),
]

type LossType = Annotated[
    BinaryCrossEntropyLossFactory,
    Field(discriminator="object_type", description="Loss configuration field."),
]

type NameType = Annotated[
    str,
    Field(
        "keras_sequential",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        description="Name of the model. Must be a valid Python identifier.",
    ),
]

type JitCompileType = Annotated[
    Literal["auto"] | bool,
    Field(
        "auto",
        description="Whether to use JIT compilation for the model.",
    ),
]

type LayerType = Annotated[
    (
        DenseFactory
        | BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory
        | StronglyEntanglingQuantumLayerWithAnglerEmbeddingFactory
        | HardwareEfficientQuantumLayerWithAnglerEmbeddingFactory
    ),
    Field(discriminator="object_type", description="Layer configuration field."),
]

type LayersType = Annotated[
    list[LayerType],
    Field(..., min_items=1, description="List of model layers."),
]


class KerasSequentialModelFactory(BaseModel):
    layers: LayersType
    name: NameType

    def as_keras(self):
        from keras import Sequential

        layers = [layer.as_keras() for layer in self.layers]
        return Sequential(layers, name=self.name)


type StandardReturnDictValues = JsonValue | datetime


class StandardEvaluationDict(TypedDict, extra_items=StandardReturnDictValues):
    loss: float
    start: datetime
    end: datetime


class StandardFitDict(StandardEvaluationDict):
    loss: list[float]

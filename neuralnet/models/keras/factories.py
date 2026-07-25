from pydantic import BaseModel, Field
from typing import Annotated, Literal, TypedDict
from datetime import datetime

from ..dense import DenseFactory
from ...layers.quantum import (
    BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory,
    StronglyEntanglingQuantumLayerWithAnglerEmbeddingFactory,
    HardwareEfficientQuantumLayerWithAnglerEmbeddingFactory,
)
from ..kan import KAN1DLayerFactory
from ...losses.binary_loss import BinaryCrossEntropyLossFactory
from ...optimizers.adam import AdamFactory

EpochsType = Annotated[
    int,
    Field(description="Number of epochs to train the model.", gt=0),
]

type VerboseType = Annotated[
    int,
    Field(ge=0, le=2, description="Verbosity of the training."),
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
        | KAN1DLayerFactory
    ),
    Field(discriminator="object_type", description="Layer configuration field."),
]

type LayersType = Annotated[
    list[LayerType],
    Field(..., min_items=1, description="List of model layers."),
]


class KerasSequentialModelFactory(BaseModel):
    layers: LayersType
    name: NameType = "keras_sequential"
    object_type: Annotated[
        Literal["keras_sequential"],
        Field(
            description="Discriminator field to identify the model type.",
        )
    ] = "keras_sequential"

    def as_keras(self):
        from keras import Sequential

        layers = [layer.as_keras() for layer in self.layers]
        return Sequential(layers, name=self.name)


class StandardEvaluationDict(TypedDict):
    loss: float
    start: datetime
    end: datetime


class FitDict(TypedDict):
    loss: list[float]


class FitRoutineDict(TypedDict):
    train: FitDict
    val: FitDict
    start: datetime
    end: datetime

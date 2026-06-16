from pydantic import BaseModel, Field, ConfigDict, JsonValue
from typing import Annotated, Literal, TypedDict
from datetime import datetime

from ..interfaces.keras import KerasFactory

type NameType = Annotated[
    str,
    Field(
        "keras_sequential",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        description="Name of the model. Must be a valid Python identifier.",
    ),
]

type CallbacksType = Annotated[
    list[KerasFactory],
    Field(
        default_factory=list,
        description="List of Keras callbacks to use during training.",
    ),
]

type JitCompileType = Annotated[
    Literal["auto"] | bool,
    Field(
        "auto",
        description="Whether to use JIT compilation for the model.",
    ),
]

type EpochsType = Annotated[
    int,
    Field(5000, description="Number of epochs to train the model.", gt=0),
]

type VerboseType = Annotated[
    int,
    Field(1, ge=0, le=2, description="Verbosity of the training."),
]

type PatienceType = Annotated[
    int,
    Field(
        25,
        description="Number of epochs with no improvement after which training will be stopped.",
    ),
]

type LayersType = Annotated[
    list[KerasFactory],
    Field(..., min_items=1, description="List of sequential model layers."),
]


class KerasModelFactory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    object_type: Literal["keras"] = Field(
        "keras",
        description='Kind of model. Must be "keras"',
    )
    name: NameType
    callbacks: CallbacksType
    loss: KerasFactory
    optimizer: KerasFactory
    jit_compile: JitCompileType

    # Training related fields
    epochs: EpochsType
    verbose: VerboseType
    patience: PatienceType

    def as_keras(self):
        from .keras import KerasModel

        return KerasModel.from_factory(self)


class KerasSequentialModelFactory(KerasModelFactory):
    layers: LayersType

    def as_keras(self):
        from .keras import KerasSequentialModel

        return KerasSequentialModel.from_factory(self)


type StandardReturnDictValues = JsonValue | datetime


class StandardEvaluationDict(TypedDict, extra_items=StandardReturnDictValues):
    loss: float
    start: datetime
    end: datetime


class StandardFitDict(StandardEvaluationDict):
    loss: list[float]

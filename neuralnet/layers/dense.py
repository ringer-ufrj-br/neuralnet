from typing import Literal
from pydantic import Field, ConfigDict

from ..interfaces.keras import KerasFactory


class DenseFactory(KerasFactory):
    model_config = ConfigDict(extra="forbid", frozen=True)

    units: int
    activation: str
    kernel_initializer: str = Field(default="glorot_uniform")
    object_type: Literal["dense"] = Field(
        default="dense", description='Layer name. Must be "dense"'
    )
    bias_initializer: str = Field(default="zeros")

    def as_keras(self):
        from keras.layers import Dense
        return Dense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

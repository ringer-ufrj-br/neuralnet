from typing import Literal
from pydantic import Field, ConfigDict, BaseModel


class DenseFactory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    units: int = Field(
        ...,
        description="Number of neurons in the dense layer.",
    )
    activation: str = Field(
        ...,
        description="Activation function for the dense layer.",
    )
    kernel_initializer: str = Field(
        "glorot_uniform",
        description="Initializer for the kernel weights matrix.",
    )
    object_type: Literal["dense"] = Field(
        "dense", description='Layer name. Must be "dense"'
    )
    bias_initializer: str = Field(
        "zeros",
        description="Initializer for the bias vector.",
    )

    def as_keras(self):
        from keras.layers import Dense

        return Dense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )


class MLPFactory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layers: list[DenseFactory] = Field(
        ...,
        description="List of dense layers in the MLP.",
    )
    name: str = Field("mlp", description="Model name")
    object_type: Literal["mlp"] = Field(
        "mlp", description='MLPFactory identifier. Must be "mlp".'
    )

    def as_keras(self):
        from keras import Sequential

        return Sequential(
            [layer.as_keras() for layer in self.layers],
            name=self.name,
        )

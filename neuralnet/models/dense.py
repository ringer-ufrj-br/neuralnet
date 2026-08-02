"""Factory models for building dense Keras components from validated configs.

This module groups Pydantic-based factories used to create dense layers and
Sequential MLP architectures.
"""

from typing import Annotated, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ..quantization.keras import FixedPointQuantizationDense
from pydantic import Field, ConfigDict, BaseModel

from ..quantization.hgq import HGQFixedPointConfig


type ActivationField = Annotated[
    str,
    Field(
        ...,
        description="Activation function for the dense layer.",
    ),
]

type BiasInitializerField = Annotated[
    str,
    Field(
        description="Initializer for the bias vector.",
    ),
]

type KernelInitializerField = Annotated[
    str,
    Field(
        description="Initializer for the kernel weights matrix.",
    ),
]

type LayerNameField = Annotated[
    str | None,
    Field(
        ...,
        description="Name of the layer. Must be a valid Python identifier. If not provided, a default name will be generated.",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    ),
]

type UnitsField = Annotated[
    int,
    Field(
        ...,
        description="Number of neurons in the dense layer.",
        gt=0,
    ),
]


class DenseFactory(BaseModel):
    """Pydantic factory for a single Keras ``Dense`` layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    units: UnitsField
    activation: ActivationField
    kernel_initializer: KernelInitializerField = "glorot_uniform"
    object_type: Literal["dense"] = Field("dense", description='Layer name. Must be "dense"')
    bias_initializer: BiasInitializerField = "zeros"
    name: LayerNameField = None

    def as_keras(self):
        """Build a ``keras.layers.Dense`` instance from this factory.

        Returns
        -------
        keras.layers.Dense
            Dense layer configured with this factory's parameters.
        """

        from keras.layers import Dense

        return Dense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=self.name,
        )


class MLPFactory(BaseModel):
    """Pydantic factory for a Sequential multi-layer perceptron model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layers: list[DenseFactory] = Field(
        ...,
        description="List of dense layers in the MLP.",
    )
    name: str = Field("mlp", description="Model name")
    object_type: Literal["mlp"] = Field("mlp", description='MLPFactory identifier. Must be "mlp".')

    def as_keras(self):
        """Build a ``keras.Sequential`` MLP model from this factory.

        Returns
        -------
        keras.Sequential
            Sequential model containing each configured dense layer in order.
        """

        from keras import Sequential

        return Sequential(
            [layer.as_keras() for layer in self.layers],
            name=self.name,
        )


class HGQ2DenseLayer(DenseFactory):
    """Pydantic factory for a single HGQ2 dense layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_type: Annotated[Literal["hgq2_dense"], Field(description='Layer name. Must be "hgq2_dense"')] = "hgq2_dense"

    quantization: HGQFixedPointConfig = Field(
        ...,
        description="Fixed-point quantization configuration for the layer.",
    )

    def as_keras(self):
        """Build a ``hgq.layers.QDense`` instance from this factory.

        Returns
        -------
        hgq.layers.QDense
            Quantized dense layer configured with this factory's parameters.
        """

        from hgq.layers import QDense

        return QDense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=self.name,
            kq_conf=self.quantization.as_hgq_quantizer_config(place="weight"),
            bq_conf=self.quantization.as_hgq_quantizer_config(place="bias"),
        )


class HGQ2FixedPointQuantizedMLP(BaseModel):
    """Pydantic factory for a Sequential multi-layer perceptron model with fixed-point quantization."""

    layers: list[HGQ2DenseLayer] = Field(
        ...,
        description="List of dense layers in the MLP.",
    )
    name: Annotated[str, Field(description="Model name")] = "hgq2_mlp"
    object_type: Annotated[Literal["hgq2_mlp"], Field(description='Object identifier. Must be "hgq2_mlp".')] = (
        "hgq2_mlp"
    )

    def as_keras(self):
        """Build a ``keras.Sequential`` MLP model from this factory.

        Returns
        -------
        keras.Sequential
            Sequential model containing each configured dense layer in order.
        """

        from keras import Sequential

        return Sequential(
            [layer.as_keras() for layer in self.layers],
            name=self.name,
        )


class FixedPointQuantizationDenseLayer(BaseModel):
    """Pydantic factory for a single Keras ``FixedPointQuantizationDense`` layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    units: UnitsField
    activation: ActivationField
    kernel_initializer: KernelInitializerField = "glorot_uniform"
    object_type: Literal["fixed_point_quantized_dense"] = Field(
        "fixed_point_quantized_dense", description='Layer name. Must be "dense"'
    )
    bias_initializer: BiasInitializerField = "zeros"
    floating_bits: int = Field(
        ...,
        description="Number of bits for the floating point part.",
        gt=0,
    )
    integer_bits: int = Field(
        ...,
        description="Number of bits for the integer part.",
        gt=0,
    )
    name: LayerNameField = None

    def as_keras(self) -> "FixedPointQuantizationDense":
        """Build a ``FixedPointQuantizationDense`` instance from this factory.

        Returns
        -------
        FixedPointQuantizationDense
            Quantized dense layer configured with this factory's parameters.
        """

        from ..quantization.keras import FixedPointQuantizationDense

        return FixedPointQuantizationDense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            floating_bits=self.floating_bits,
            integer_bits=self.integer_bits,
            name=self.name,
        )


class FixedPointQuantizedMLP(BaseModel):
    """Pydantic factory for a Sequential multi-layer perceptron model with fixed-point quantization."""

    layers: list[FixedPointQuantizationDenseLayer] = Field(
        ...,
        description="List of dense layers in the MLP.",
    )
    name: Annotated[str, Field(description="Model name")] = "fixed_point_quantized_mlp"
    object_type: Annotated[
        Literal["fixed_point_quantized_mlp"],
        Field(description='Object identifier. Must be "fixed_point_quantized_mlp".'),
    ] = "fixed_point_quantized_mlp"

    def as_keras(self):
        """Build a ``keras.Sequential`` MLP model from this factory.

        Returns
        -------
        keras.Sequential
            Sequential model containing each configured dense layer in order.
        """

        from keras import Sequential

        return Sequential(
            [layer.as_keras() for layer in self.layers],
            name=self.name,
        )

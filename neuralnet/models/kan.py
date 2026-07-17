"""Factory models for building KAN components from validated configs.

This module groups Pydantic-based factories used to create KAN layers and KAN
models for PyTorch and Keras integration.
"""

from typing import Annotated, Annotated, Literal
from pydantic import BaseModel, Field


type GridSizeType = Annotated[
    int,
    Field(
        ...,
        description="The size of the grid for the KAN.",
        gt=0,
    ),
]

type SplineOrderType = Annotated[
    int,
    Field(
        ...,
        description="The order of the spline used for the KAN.",
        gt=0,
    ),
]


class KAN1DLayerFactory(BaseModel):
    """Pydantic factory for a single 1D KAN layer."""

    input_size: int = Field(..., description="The input size.", gt=0)
    output_size: int = Field(..., description="The output size.", gt=0)
    grid_size: GridSizeType
    spline_order: SplineOrderType
    name: str | None = Field(
        None,
        description="The name of the layer.",
    )

    object_type: Literal["kan1d_layer"] = Field(
        "kan1d_layer",
        description="KAN 1D layer identifier.",
    )

    def as_torch(self):
        """Build an ``efficient_kan.KAN`` layer for PyTorch.

        Returns
        -------
        efficient_kan.KAN
            KAN module configured as a single layer transform.

        Raises
        ------
        ValueError
            If ``name`` is provided, since naming is not supported for the
            generated PyTorch layer.
        """

        from efficient_kan import KAN

        if self.name is not None:
            raise ValueError("Naming is not supported for PyTorch layers.")

        return KAN(
            [self.input_size, self.output_size],
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )

    def as_keras(self):
        """Build a Keras wrapper for the generated PyTorch KAN layer.

        Returns
        -------
        keras.layers.TorchModuleWrapper
            Keras layer wrapping the underlying PyTorch KAN module.

        Raises
        ------
        ValueError
            If the active Keras backend is not ``torch``.
        """

        import keras
        from keras.layers import TorchModuleWrapper

        backend = keras.config.backend()
        if backend != "torch":
            raise ValueError(
                f"Invalid Keras backend '{backend}'. Quantum layers require the 'torch' backend."
            )
        torch_layer = self.as_torch()
        return TorchModuleWrapper(
            torch_layer,
            output_shape=[None, self.output_size],
            name=self.name,
        )


type ModelShapeElementType = Annotated[
    int,
    Field(
        gt=0,
    ),
]


class KANModelFactory(BaseModel):
    """Pydantic factory for a multi-layer 1D KAN model."""

    model_shape: list[ModelShapeElementType] = Field(
        ...,
        description="The shape of the KAN model, defined as a list of layer sizes transformations. It should start with the input size and end with the output size.",
    )
    grid_size: GridSizeType
    spline_order: SplineOrderType
    name: str | None = Field(
        None,
        description="The name of the layer.",
    )

    object_type: Literal["kan1d"] = Field(
        "kan1d",
        description="KAN 1D layer identifier.",
    )

    object_type: Literal["kan_model"] = Field(
        "kan_model",
        description="KAN model identifier.",
    )

    def as_torch(self):
        """Build an ``efficient_kan.KAN`` model for PyTorch.

        Returns
        -------
        efficient_kan.KAN
            KAN module configured from the full model shape.

        Raises
        ------
        ValueError
            If ``name`` is provided, since naming is not supported for the
            generated PyTorch layer.
        """

        from efficient_kan import KAN

        if self.name is not None:
            raise ValueError("Naming is not supported for PyTorch layers.")

        return KAN(
            self.model_shape,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )

    def as_keras(self):
        """Build a Keras model that wraps the generated PyTorch KAN module.

        Returns
        -------
        keras.Model
            Keras model exposing the wrapped KAN module as its output.

        Raises
        ------
        ValueError
            If the active Keras backend is not ``torch``.
        """

        from keras.config import backend
        from keras.layers import TorchModuleWrapper
        from keras import Input, Model

        if backend() != "torch":
            raise ValueError(
                f"Invalid Keras backend '{backend}'. Quantum layers require the 'torch' backend."
            )
        torch_layer = self.as_torch()
        input_layer = Input(shape=(self.model_shape[0],))
        kan_layer = TorchModuleWrapper(
            torch_layer, output_shape=[None, self.model_shape[-1]]
        )
        model = Model(inputs=input_layer, outputs=kan_layer, name=self.name)
        return model

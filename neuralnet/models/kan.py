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
        from efficient_kan import KAN

        if self.name is not None:
            raise ValueError("Naming is not supported for PyTorch layers.")

        return KAN(
            [self.input_size, self.output_size],
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )

    def as_keras(self):
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
        from efficient_kan import KAN

        if self.name is not None:
            raise ValueError("Naming is not supported for PyTorch layers.")

        return KAN(
            self.model_shape,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )

    def as_keras(self):
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

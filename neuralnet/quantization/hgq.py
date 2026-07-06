from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Literal, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from keras import Sequential
    from hgq.config import QuantizerConfig


type PlaceType = Literal["kernel", "bias"]


class HGQFixedPointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    i0: Annotated[
        int,
        Field(
            description="Number of integer bits for the fixed-point representation.",
        ),
    ]
    f0: Annotated[
        int,
        Field(
            description="Number of fractional bits for the fixed-point representation.",
        ),
    ]

    def as_hgq_quantizer_config(self, place: PlaceType):
        from hgq.config import QuantizerConfig
        from hgq.constraints import Constant

        return QuantizerConfig(
            q_type="kif",
            place=place,
            k0=True,
            i0=self.i0,
            f0=self.f0,
            fc=Constant(self.f0),
            ic=Constant(self.i0),
        )


def hgq_quantize(
    model: "Sequential",
    weight_quantizer_config: Union[HGQFixedPointConfig, 'QuantizerConfig'],
    bias_quantizer_config: Union[HGQFixedPointConfig, 'QuantizerConfig'],
) -> "Sequential":
    from ..models.dense.hgq import keras_dense_to_hgq_dense
    from keras import Sequential, Input
    from keras.layers import Dense

    if not isinstance(model, Sequential):
        raise TypeError(f"Expected keras.Sequential model, got {type(model)}")
    
    if isinstance(weight_quantizer_config, HGQFixedPointConfig):
        weight_quantizer_config = weight_quantizer_config.as_hgq_quantizer_config(place="weight")

    if isinstance(bias_quantizer_config, HGQFixedPointConfig):
        bias_quantizer_config = bias_quantizer_config.as_hgq_quantizer_config(place="bias")

    quantized_layers = [
        Input(shape=model.input_shape[1:]),
    ]
    for layer in model.layers:
        if isinstance(layer, Dense):
            quantized_layer = keras_dense_to_hgq_dense(
                layer,
                kq_conf=weight_quantizer_config,
                bq_conf=bias_quantizer_config
            )
            quantized_layers.append(quantized_layer)
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

    quantized_model = Sequential(quantized_layers, name=f"{model.name}_quantized")

    return quantized_model

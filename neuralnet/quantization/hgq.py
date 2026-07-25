"""Quantization utilities backed by the hgq2 library.

This module contains helpers for converting Keras Sequential models into
HGQ-quantized models using hgq2 as the backend implementation.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Literal, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from keras import Sequential
    from hgq.config import QuantizerConfig
    from hgq.layers import QDense
    from hgq.config import QuantizerConfig
    from keras.layers import Dense


class HGQFixedPointConfig(BaseModel):
    """Fixed-point quantizer configuration.

    Parameters
    ----------
    i0 : int
        Number of integer bits for the fixed-point representation.
    f0 : int
        Number of fractional bits for the fixed-point representation.
    """

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

    def as_hgq_quantizer_config(self, place: Literal["kernel", "bias"]):
        """Create an hgq2 quantizer configuration from fixed-point settings.

        Parameters
        ----------
        place : {"kernel", "bias"}
            Target tensor place for the quantizer configuration.

        Returns
        -------
        QuantizerConfig
            An hgq2 quantizer configuration using constant integer and
            fractional bit settings.
        """
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
    weight_quantizer_config: Union[HGQFixedPointConfig, "QuantizerConfig"],
    bias_quantizer_config: Union[HGQFixedPointConfig, "QuantizerConfig"],
    name: str | None = None,
) -> "Sequential":
    """Quantize a Keras Sequential model using hgq2 backend layers.
    Currently only support models with Dense layers

    Parameters
    ----------
    model : keras.Sequential
        Source Sequential model to convert.
    weight_quantizer_config : HGQFixedPointConfig or QuantizerConfig
        Quantizer configuration for the dense layer kernel weights.
    bias_quantizer_config : HGQFixedPointConfig or QuantizerConfig
        Quantizer configuration for the dense layer biases.
    name : str, optional
        Name for the quantized Sequential model. If omitted, the new model name will
        be "{old_name}_quantized"

    Returns
    -------
    keras.Sequential
        A new Sequential model whose Dense layers have been replaced by
        hgq2 QDense Layers.

    Raises
    ------
    TypeError
        If ``model`` is not a Keras Sequential instance or contains an
        unsupported layer type.
    """
    from keras import Sequential, Input
    from keras.layers import Dense

    if not isinstance(model, Sequential):
        raise TypeError(f"Expected keras.Sequential model, got {type(model)}")

    if isinstance(weight_quantizer_config, HGQFixedPointConfig):
        weight_quantizer_config = weight_quantizer_config.as_hgq_quantizer_config(
            place="weight"
        )

    if isinstance(bias_quantizer_config, HGQFixedPointConfig):
        bias_quantizer_config = bias_quantizer_config.as_hgq_quantizer_config(
            place="bias"
        )

    quantized_layers = [
        Input(shape=model.input_shape[1:]),
    ]
    for layer in model.layers:
        if isinstance(layer, Dense):
            quantized_layer = keras_dense_to_hgq_dense(
                layer, kq_conf=weight_quantizer_config, bq_conf=bias_quantizer_config
            )
            quantized_layers.append(quantized_layer)
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

    if name is None:
        name = f"{model.name}_quantized"
    quantized_model = Sequential(quantized_layers, name=name)

    return quantized_model


def keras_dense_to_hgq_dense(
    keras_layer: 'Dense', kq_conf: 'QuantizerConfig' = None, bq_conf: 'QuantizerConfig' = None
) -> 'QDense':
    """Convert a Keras Dense layer into an HGQ quantized dense layer.

    Parameters
    ----------
    keras_layer : keras.layers.Dense
        Source Keras dense layer whose configuration and weights will be copied.
    kq_conf : QuantizerConfig, optional
        Configuration for the kernel quantizer.
    bq_conf : QuantizerConfig, optional
        Configuration for the bias quantizer.

    Returns
    -------
    hgq.layers.QDense
        A quantized dense layer with the same configuration and copied weights.
    """
    from hgq.layers import QDense
    quantized_layer = QDense(
        units=keras_layer.units,
        activation=keras_layer.activation,
        use_bias=keras_layer.use_bias,
        kernel_initializer=keras_layer.kernel_initializer,
        bias_initializer=keras_layer.bias_initializer,
        kernel_regularizer=keras_layer.kernel_regularizer,
        bias_regularizer=keras_layer.bias_regularizer,
        activity_regularizer=keras_layer.activity_regularizer,
        kernel_constraint=keras_layer.kernel_constraint,
        bias_constraint=keras_layer.bias_constraint,
        kq_conf=kq_conf,
        bq_conf=bq_conf,
    )
    quantized_layer.build((None, int(keras_layer.kernel.shape[0])))
    quantized_layer_weights = quantized_layer.get_weights()
    keras_weights = keras_layer.get_weights()
    quantized_layer_weights[0] = keras_weights[0]  # kernel
    if len(keras_weights) > 1:
        quantized_layer_weights[1] = keras_weights[1]  # bias
    quantized_layer.set_weights(quantized_layer_weights)
    return quantized_layer

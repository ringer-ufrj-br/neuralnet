"""Custom fixed-point quantization utilities implemented without hgq2.

This module contains the project's bespoke Keras-based quantization layers and
helpers for converting Sequential models into fixed-point quantized versions.
"""

from keras import activations, ops, initializers, Sequential, saving
from keras.layers import Layer, Dense, Activation


@ops.custom_gradient
def fixed_point_quantizer(x, integer_bits: int, floating_bits: int):
    """Quantize values to a fixed-point grid with straight-through gradients.

    Parameters
    ----------
    x : Tensor-like
        Input values to quantize.
    integer_bits : int
        Number of integer bits used to define the representable range.
    floating_bits : int
        Number of fractional bits used to define the quantization step.

    Returns
    -------
    Tensor-like
        Quantized values clipped to the representable fixed-point range.
    """
    # Forward pass: quantize the input
    # saturated_x = clip(x, lower_bound, upper_bound)
    floating_power = 2**floating_bits
    lower_bound = -(2**integer_bits)
    upper_bound = (2**integer_bits) - (2 ** (-floating_bits))
    int_x = ops.round(x * floating_power)
    quantized_x = int_x / floating_power
    clipped_quantized_x = ops.clip(quantized_x, lower_bound, upper_bound)

    # Backward pass computation
    def grad(*args, upstream=None):
        if upstream is None:
            upstream = args[0]  # PyTorch sometimes puts args in a tuple
        return upstream, None, None

    return clipped_quantized_x, grad


@saving.register_keras_serializable(package="neuralnet")
class FixedPointQuantizationDense(Layer):
    """Dense layer that applies custom fixed-point quantization.

    Attributes
    ----------
    units : int
        Number of output units produced by the dense layer.
    activation : str or callable or None
        Activation identifier used to configure the post-affine activation.
    kernel_initializer : str or initializers.Initializer
        Initializer for the kernel weight matrix.
    bias_initializer : str or initializers.Initializer
        Initializer for the bias vector.
    kernel : keras.Variable
        Trainable weight matrix for the dense transform.
    bias : keras.Variable
        Trainable bias vector added after the matrix multiplication.
    activation_fn : callable or None
        Resolved activation function, or ``None`` when no activation is set.
    """

    def __init__(
        self,
        floating_bits: int,
        integer_bits: int,
        units: int,
        activation: str | None = None,
        kernel_initializer: str | initializers.Initializer = "glorot_uniform",
        bias_initializer: str | initializers.Initializer = "zeros",
        **kwargs,
    ):
        """Initialize a quantized dense layer.

        Parameters
        ----------
        floating_bits : int
            Number of fractional bits used by the quantizer.
        integer_bits : int
            Number of integer bits used by the quantizer.
        units : int
            Number of output units in the dense layer.
        activation : str or callable, optional
            Activation to apply after the quantized dense computation. If not
            provided, the layer is linear.
        kernel_initializer : str or initializers.Initializer, optional
            Initializer for the kernel weight matrix. Defaults to ``"glorot_uniform"``.
        bias_initializer : str or initializers.Initializer, optional
            Initializer for the bias vector. Defaults to ``"zeros"``.
        **kwargs
            Additional keyword arguments forwarded to the base layer.
        """
        super().__init__(**kwargs)
        self._floating_bits_val = int(floating_bits)
        self._integer_bits_val = int(integer_bits)
        # Casting so keras can make the bits operation with float32 tensors
        self.floating_bits = self.add_weight(
            name="floating_bits",
            shape=(),
            initializer=initializers.Constant(floating_bits),
            trainable=False,
        )
        self.integer_bits = self.add_weight(
            name="integer_bits",
            shape=(),
            initializer=initializers.Constant(integer_bits),
            trainable=False,
        )
        self.units = units
        self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        """Create the trainable kernel and bias weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the incoming tensor, used to infer the kernel shape.
        """
        # Create trainable weights for the layer
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer=self.bias_initializer,
            trainable=True,
        )
        if self.activation is not None:
            self.activation_fn = activations.get(self.activation)
        else:
            self.activation_fn = None

    def call(self, inputs):
        """Apply quantized dense computation to the input tensor.

        Parameters
        ----------
        inputs : Tensor-like
            Input activations to transform.

        Returns
        -------
        Tensor-like
            Output activations after quantized matrix multiplication, bias
            addition, and optional activation.
        """
        # Apply quantization to the kernel weights
        quantized_kernel = fixed_point_quantizer(self.kernel, self.integer_bits, self.floating_bits)
        quantized_bias = fixed_point_quantizer(self.bias, self.integer_bits, self.floating_bits)
        matmul = ops.matmul(inputs, quantized_kernel)
        quantized_matmul = fixed_point_quantizer(matmul, self.integer_bits, self.floating_bits)
        added_bias = ops.add(quantized_matmul, quantized_bias)
        added_bias = fixed_point_quantizer(added_bias, self.integer_bits, self.floating_bits)
        if self.activation_fn is not None:
            return self.activation_fn(added_bias)
        return added_bias

    def get_config(self):
        """Return layer configuration dictionary for serialization.

        Returns
        -------
        dict
            Layer configuration dictionary containing bit parameters and layer dimensions.
        """
        config = super().get_config()
        config.update(
            {
                "floating_bits": self._floating_bits_val,
                "integer_bits": self._integer_bits_val,
                "units": self.units,
                "activation": self.activation,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
            }
        )
        return config


type SuportedFixedPointQuantizationTypes = Dense | Activation

type FixedPointQuantizationLayerTypes = FixedPointQuantizationDense | Activation


def fixed_point_quantize_layer(
    layer: SuportedFixedPointQuantizationTypes, floating_bits: int, integer_bits: int
) -> FixedPointQuantizationLayerTypes:
    """Convert a supported Keras layer into a fixed-point quantized layer.

    Parameters
    ----------
    layer : Dense or Activation
        Layer to convert.
    floating_bits : int
        Number of fractional bits used by the quantizer.
    integer_bits : int
        Number of integer bits used by the quantizer.

    Returns
    -------
    FixedPointQuantizationDense or Activation
        Quantized replacement for the input layer.

    Raises
    ------
    NotImplementedError
        If the layer type is not supported.
    """
    if isinstance(layer, Dense):
        # if layer.use_bias:
        #     raise NotImplementedError(
        #         "Fixed-point quantization for Dense layers with bias is not implemented yet.")
        quantized_layer = FixedPointQuantizationDense(
            floating_bits=floating_bits,
            integer_bits=integer_bits,
            units=layer.units,
            activation=layer.activation,
            kernel_initializer=layer.kernel_initializer,
            bias_initializer=layer.bias_initializer,
            name=f"quantized_{layer.name}",
        )
        quantized_layer.build((None, int(layer.kernel.shape[0])))
        fp_weights = layer.get_weights()
        quantized_weights = quantized_layer.get_weights()
        quantized_weights[0] = fp_weights[0]  # Copy kernel weights
        if len(fp_weights) > 1 and len(quantized_weights) > 1:
            quantized_weights[1] = fp_weights[1]  # Copy bias weights
        quantized_layer.set_weights(quantized_weights)
    elif isinstance(layer, Activation):
        # For activation layers, we don't quantize them directly
        return layer
    else:
        raise NotImplementedError(f"Unsupported layer type for fixed-point quantization: {type(layer)}")

    return quantized_layer


def fixed_point_quantize(
    model: "Sequential", floating_bits: int, integer_bits: int, name: str | None = None
) -> "Sequential":
    """Quantize all supported layers in a Keras Sequential model.

    Parameters
    ----------
    model : keras.Sequential
        Source model to convert.
    floating_bits : int
        Number of fractional bits used by the quantizer.
    integer_bits : int
        Number of integer bits used by the quantizer.
    name : str, optional
        Name for the quantized model. If omitted, a name is derived from the
        source model.

    Returns
    -------
    keras.Sequential
        A new Sequential model built from fixed-point quantized layers.
    """
    from keras import Sequential, Input

    quantized_layers = [
        Input(shape=model.input_shape[1:]),
    ]
    for layer in model.layers:
        quantized_layer = fixed_point_quantize_layer(layer, floating_bits=floating_bits, integer_bits=integer_bits)
        quantized_layers.append(quantized_layer)
    if name is None:
        name = f"{model.name}_quantized"
    return Sequential(quantized_layers, name=name)

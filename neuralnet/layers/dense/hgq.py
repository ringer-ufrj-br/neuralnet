from hgq.layers import QDense
from hgq.config import QuantizerConfig
from keras.layers import Dense


def keras_dense_to_hgq_dense(
    keras_layer: Dense, kq_conf: QuantizerConfig = None, bq_conf: QuantizerConfig = None
):
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

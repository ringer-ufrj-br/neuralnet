import os

import numpy as np


def test_hgqdense_from_keras_dense_copies_configuration_and_weights(isolated_executor):
    future = isolated_executor.submit(
        hgqdense_from_keras_dense_copies_configuration_and_weights,
    )
    future.result()


def hgqdense_from_keras_dense_copies_configuration_and_weights():
    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras
    from hgq.layers import QDense
    from neuralnet.quantization.hgq import keras_dense_to_hgq_dense

    keras_layer = keras.layers.Dense(
        units=3,
        activation="relu",
        use_bias=True,
        kernel_initializer="zeros",
        bias_initializer="zeros",
    )
    keras_layer.build((None, 4))
    expected_weights = [
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
    ]
    keras_layer.set_weights(expected_weights)

    quantized_layer = keras_dense_to_hgq_dense(keras_layer)

    assert isinstance(quantized_layer, QDense)
    assert quantized_layer.units == keras_layer.units
    assert quantized_layer.activation == keras_layer.activation
    assert quantized_layer.use_bias is True
    quantized_weights = quantized_layer.get_weights()
    np.testing.assert_allclose(quantized_weights[0], expected_weights[0])
    np.testing.assert_allclose(quantized_weights[1], expected_weights[1])

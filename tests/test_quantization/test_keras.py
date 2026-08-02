from pathlib import Path


def test_fixed_point_quantization_dense_training_and_serialization(
    tmp_path: Path,
    isolated_executor,
):
    """Test training, evaluation, saving, loading, and prediction with FixedPointQuantizationDense."""
    future = isolated_executor.submit(
        _fixed_point_quantization_dense_routine,
        tmp_path=tmp_path,
    )
    future.result()


def _fixed_point_quantization_dense_routine(tmp_path: Path):
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras
    import numpy as np
    from sklearn.datasets import make_blobs
    from neuralnet.quantization.keras import (
        FixedPointQuantizationDense,
    )

    # 1. Create dataset for an easy binary classification problem
    X, y = make_blobs(n_samples=300, centers=2, n_features=4, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # 2. Build Keras Sequential model with FixedPointQuantizationDense layers
    model = keras.Sequential(
        [
            keras.Input(shape=(4,)),
            FixedPointQuantizationDense(
                floating_bits=8, integer_bits=4, units=8, activation="relu"
            ),
            FixedPointQuantizationDense(
                floating_bits=8, integer_bits=4, units=1, activation="sigmoid"
            ),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 3. Run training
    model.fit(X, y, epochs=40, batch_size=32, verbose=0)

    # 4. Check if metrics are good
    loss, accuracy = model.evaluate(X, y, verbose=0)
    assert accuracy > 0.90, f"Expected accuracy > 0.90, got {accuracy}"

    # 5. Save model to a file
    model_path = tmp_path / "quantized_dense_model.keras"
    model.save(model_path)
    assert model_path.exists(), "Model file was not saved successfully"

    # 6. Load model again
    loaded_model = keras.models.load_model(model_path)

    # 7. Run predictions to check if loading and saving works
    preds_orig = model.predict(X[:10], verbose=0)
    preds_loaded = loaded_model.predict(X[:10], verbose=0)

    np.testing.assert_allclose(
        preds_orig,
        preds_loaded,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Predictions from original and loaded models do not match",
    )


def test_fixed_point_quantize_sequential_model(
    tmp_path: Path,
    isolated_executor,
):
    """Test fixed_point_quantize helper converting a standard Sequential model."""
    future = isolated_executor.submit(
        _fixed_point_quantize_sequential_routine,
        tmp_path=tmp_path,
    )
    future.result()


def _fixed_point_quantize_sequential_routine(tmp_path: Path):
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras
    import numpy as np
    from sklearn.datasets import make_blobs
    from neuralnet.quantization.keras import (
        fixed_point_quantize,
    )

    X, y = make_blobs(n_samples=300, centers=2, n_features=4, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    base_model = keras.Sequential(
        [
            keras.Input(shape=(4,)),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    base_model.build((None, 4))

    # Verify that with a large number of bits, floating-point and quantized outputs are close
    fp_preds_before_fit = base_model.predict(X[:20], verbose=0)
    high_precision_quant_model = fixed_point_quantize(
        base_model, floating_bits=16, integer_bits=16
    )
    quant_preds_before_fit = high_precision_quant_model.predict(X[:20], verbose=0)

    np.testing.assert_allclose(
        fp_preds_before_fit,
        quant_preds_before_fit,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Outputs of floating point model and high-precision quantized model are not close enough",
    )


def test_fixed_point_quantization_dense_initializers(isolated_executor):
    """Test custom kernel_initializer and bias_initializer on FixedPointQuantizationDense."""
    future = isolated_executor.submit(_fixed_point_quantization_dense_initializers_routine)
    future.result()


def _fixed_point_quantization_dense_initializers_routine():
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    import numpy as np
    from neuralnet.quantization.keras import FixedPointQuantizationDense

    layer = FixedPointQuantizationDense(
        floating_bits=8,
        integer_bits=4,
        units=5,
        kernel_initializer="ones",
        bias_initializer="ones",
    )
    layer.build((None, 3))

    np.testing.assert_allclose(layer.get_weights()[0], np.ones((3, 5), dtype=np.float32))
    np.testing.assert_allclose(layer.get_weights()[1], np.ones((5,), dtype=np.float32))

    config = layer.get_config()
    assert config["kernel_initializer"]["class_name"] == "Ones"
    assert config["bias_initializer"]["class_name"] == "Ones"
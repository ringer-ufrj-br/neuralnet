import os

import numpy as np
import polars as pl
import pytest


def make_dataframe(data: dict[str, list[float | int]]) -> pl.DataFrame:
    return pl.DataFrame(data)


def make_lazyframe(data: dict[str, list[float | int]]) -> pl.LazyFrame:
    return pl.DataFrame(data).lazy()


def build_constant_binned_model():
    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras

    from neuralnet.models.binned_committee import BinnedModel, VariableBin

    keras_model = keras.Sequential(
        [
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(2, activation="relu", use_bias=True),
            keras.layers.Dense(1, activation="sigmoid", use_bias=True),
        ]
    )
    keras_model(np.zeros((1, 2), dtype=np.float32))
    keras_model.layers[0].set_weights(
        [
            np.zeros((2, 2), dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        ]
    )
    keras_model.layers[1].set_weights(
        [
            np.zeros((2, 1), dtype=np.float32),
            np.array([np.log(0.7 / 0.3)], dtype=np.float32),
        ]
    )
    bins = [
        VariableBin(var_name="x", lower=0.0, upper=1.0, closed="both"),
        VariableBin(var_name="y", lower=2.0, upper=3.0, closed="both"),
    ]
    return BinnedModel(
        bins=bins,
        keras_model=keras_model,
        features=["x", "y"],
        decision_threshold=0.5,
    )


@pytest.mark.parametrize("frame_factory", [make_dataframe, make_lazyframe])
def test_binned_model_predict_polars_filters_rows_and_adds_predictions(
    frame_factory, isolated_executor
):
    future = isolated_executor.submit(
        binned_model_predict_polars_filters_rows_and_adds_predictions,
        frame_factory=frame_factory,
    )
    future.result()


def binned_model_predict_polars_filters_rows_and_adds_predictions(frame_factory):
    model = build_constant_binned_model()
    data = frame_factory(
        {
            "id": [1, 2, 3, 4],
            "x": [0.5, 1.5, 0.5, 0.5],
            "y": [2.5, 2.5, 3.5, 2.5],
        }
    )

    result = model.predict_polars(data)

    assert result.columns == ["id", "prediction", "output"]
    assert result.height == 2
    assert result["id"].to_list() == [1, 4]
    assert result["prediction"].to_list() == [True, True]
    for output in result["output"].to_numpy():
        assert output == pytest.approx(0.7, rel=1e-5)


@pytest.mark.parametrize("frame_factory", [make_dataframe, make_lazyframe])
def test_binned_model_predict_polars_with_output_layers_adds_layer_outputs(
    frame_factory, isolated_executor
):
    future = isolated_executor.submit(
        binned_model_predict_polars_with_output_layers_adds_layer_outputs,
        frame_factory=frame_factory,
    )
    future.result()


def binned_model_predict_polars_with_output_layers_adds_layer_outputs(frame_factory):
    model = build_constant_binned_model()
    data = frame_factory(
        {
            "id": [1, 2, 3, 4],
            "x": [0.5, 1.5, 0.5, 0.5],
            "y": [2.5, 2.5, 3.5, 2.5],
        }
    )

    result = model.predict_polars(data, output_layers=True)

    assert result.columns == [
        "id",
        "prediction",
        "output",
        "layer.dense.0",
        "layer.dense.1",
        "layer.dense_1.0",
    ]
    assert result.height == 2
    assert result["id"].to_list() == [1, 4]
    assert result["prediction"].to_list() == [True, True]
    for output, layer_output in zip(
        result["output"].to_numpy(),
        result["layer.dense_1.0"].to_numpy(),
    ):
        assert output == pytest.approx(0.7, rel=1e-5)
        assert layer_output == pytest.approx(0.7, rel=1e-5)

    assert result["layer.dense.0"].to_list() == [1.0, 1.0]
    assert result["layer.dense.1"].to_list() == [2.0, 2.0]


def test_binned_model_quantize_returns_copy_with_quantized_keras_model(isolated_executor):
    future = isolated_executor.submit(
        binned_model_quantize_returns_copy_with_quantized_keras_model,
    )
    future.result()


def binned_model_quantize_returns_copy_with_quantized_keras_model():
    model = build_constant_binned_model()

    import neuralnet.quantization.hgq as hgq_module

    original_hgq_quantize = hgq_module.hgq_quantize

    def fake_hgq_quantize(keras_model, kq_conf=None, bq_conf=None):
        assert keras_model is model.keras_model
        assert kq_conf == "weight-config"
        assert bq_conf == "bias-config"
        return keras_model

    hgq_module.hgq_quantize = fake_hgq_quantize
    try:
        quantized = model.quantize(kq_conf="weight-config", bq_conf="bias-config")
    finally:
        hgq_module.hgq_quantize = original_hgq_quantize

    assert quantized is not model
    assert quantized.keras_model is model.keras_model
    assert quantized.bins == model.bins
    assert quantized.features == model.features
    assert quantized.decision_threshold == model.decision_threshold


@pytest.mark.parametrize("frame_factory", [make_dataframe, make_lazyframe])
def test_binned_committee_predict_concatenates_model_predictions(
    frame_factory, isolated_executor
):
    future = isolated_executor.submit(
        binned_committee_predict_concatenates_model_predictions,
        frame_factory=frame_factory,
    )
    future.result()


def binned_committee_predict_concatenates_model_predictions(frame_factory):
    from neuralnet.models.binned_committee import BinnedCommittee

    class FakeModel:
        def __init__(self, result: pl.DataFrame):
            self.result = result
            self.received_data = None

        def predict(self, data, batch_size: int = 32):
            self.received_data = data
            return self.result

    committee = BinnedCommittee.__new__(BinnedCommittee)
    committee.models = [
        FakeModel(
            pl.DataFrame(
                {
                    "id": [1],
                    "output": [0.1],
                    "prediction": [False],
                }
            )
        ),
        FakeModel(
            pl.DataFrame(
                {
                    "id": [2],
                    "output": [0.9],
                    "prediction": [True],
                }
            )
        ),
    ]

    data = frame_factory({"id": [1, 2], "x": [0.5, 0.6]})
    result = committee.predict(data)

    assert result.columns == ["id", "output", "prediction"]
    assert result["id"].to_list() == [1, 2]
    assert result["prediction"].to_list() == [False, True]
    assert committee.models[0].received_data is data
    assert committee.models[1].received_data is data


def test_binned_committee_quantize_quantizes_each_model(isolated_executor):
    future = isolated_executor.submit(
        binned_committee_quantize_quantizes_each_model,
    )
    future.result()


def binned_committee_quantize_quantizes_each_model():
    from neuralnet.models.binned_committee import BinnedCommittee

    class FakeModel:
        def __init__(self, name: str):
            self.name = name
            self.calls = []

        def quantize(self, kq_conf=None, bq_conf=None):
            self.calls.append((kq_conf, bq_conf))
            return f"{self.name}-quantized"

    committee = BinnedCommittee.__new__(BinnedCommittee)
    committee.models = [FakeModel("model-a"), FakeModel("model-b")]

    def fake_model_copy(update):
        copied = BinnedCommittee.__new__(BinnedCommittee)
        copied.models = update["models"]
        return copied

    committee.model_copy = fake_model_copy

    quantized = committee.quantize(kq_conf="weight-config", bq_conf="bias-config")

    assert quantized.models == ["model-a-quantized", "model-b-quantized"]
    assert committee.models[0].calls == [("weight-config", "bias-config")]
    assert committee.models[1].calls == [("weight-config", "bias-config")]

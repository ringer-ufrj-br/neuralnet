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
        VariableBin(var_name="x", low=0.0, high=1.0, closed="both"),
        VariableBin(var_name="y", low=2.0, high=3.0, closed="both"),
    ]
    model = BinnedModel(
        bins=bins,
        keras_model=keras_model,
        features=["x", "y"],
        decision_threshold=0.5,
    )
    return model


def collect_if_lazy(frame):
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    return frame


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

    result = collect_if_lazy(model.predict_polars(data))

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
        binned_model_predict_polars_with_all_layers_adds_layer_outputs,
        frame_factory=frame_factory,
    )
    future.result()


def binned_model_predict_polars_with_all_layers_adds_layer_outputs(frame_factory):
    model = build_constant_binned_model()
    data = frame_factory(
        {
            "id": [1, 2, 3, 4],
            "x": [0.5, 1.5, 0.5, 0.5],
            "y": [2.5, 2.5, 3.5, 2.5],
        }
    )

    result = collect_if_lazy(model.predict_polars(data, all_layers=True))

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


@pytest.mark.parametrize("frame_factory", [make_dataframe, make_lazyframe])
def test_binned_model_predict_polars_with_all_layers_and_passthrough_adds_layer_outputs(
    frame_factory, isolated_executor
):
    future = isolated_executor.submit(
        binned_model_predict_polars_with_all_layers_and_passthrough_adds_layer_outputs,
        frame_factory=frame_factory,
    )
    future.result()


def binned_model_predict_polars_with_all_layers_and_passthrough_adds_layer_outputs(
    frame_factory,
):
    model = build_constant_binned_model()
    data = frame_factory(
        {
            "id": [1, 2, 3, 4],
            "tag": [10, 20, 30, 40],
            "x": [0.5, 1.5, 0.5, 0.5],
            "y": [2.5, 2.5, 3.5, 2.5],
        }
    )

    result = collect_if_lazy(
        model.predict_polars(data, all_layers=True, passthrough=True)
    )

    assert result.columns == [
        "id",
        "tag",
        "x",
        "y",
        "prediction",
        "output",
        "layer.dense.0",
        "layer.dense.1",
        "layer.dense_1.0",
    ]
    assert result.height == 4
    assert result["id"].to_list() == [1, 2, 3, 4]
    assert result["tag"].to_list() == [10, 20, 30, 40]
    assert result["prediction"].to_list() == [True, None, None, True]
    assert result["output"].drop_nulls().to_list() == pytest.approx(
        [0.7, 0.7], rel=1e-5
    )
    assert result["layer.dense.0"].to_list() == [1.0, None, None, 1.0]
    assert result["layer.dense.1"].to_list() == [2.0, None, None, 2.0]
    assert result["layer.dense_1.0"].drop_nulls().to_list() == pytest.approx(
        [0.7, 0.7], rel=1e-5
    )


@pytest.mark.parametrize("frame_factory", [make_dataframe])
def test_binned_model_predict_polars_with_passthrough_adds_original_columns(
    frame_factory, isolated_executor
):
    future = isolated_executor.submit(
        binned_model_predict_polars_with_passthrough_adds_original_columns,
        frame_factory=frame_factory,
    )
    future.result()


def binned_model_predict_polars_with_passthrough_adds_original_columns(
    frame_factory,
):
    model = build_constant_binned_model()
    data = frame_factory(
        {
            "id": [1, 2, 3, 4],
            "tag": [10, 20, 30, 40],
            "x": [0.5, 1.5, 0.5, 0.5],
            "y": [2.5, 2.5, 3.5, 2.5],
        }
    )

    result = collect_if_lazy(model.predict_polars(data, passthrough=True))

    assert result.columns == [
        "id",
        "tag",
        "x",
        "y",
        "prediction",
        "output",
    ]
    assert result.height == 4
    assert result["id"].to_list() == [1, 2, 3, 4]
    assert result["tag"].to_list() == [10, 20, 30, 40]
    assert result["prediction"].to_list() == [True, None, None, True]
    assert result["output"].drop_nulls().to_list() == pytest.approx(
        [0.7, 0.7], rel=1e-5
    )


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

        def predict(self, data, batch_size: int = 32, all_layers: bool = False):
            self.received_data = data
            return self.result

    committee = BinnedCommittee(
        models=[
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
    )

    data = frame_factory({"id": [1, 2], "x": [0.5, 0.6]})
    result = collect_if_lazy(committee.predict(data))

    assert result.columns == ["id", "output", "prediction"]
    assert result["id"].to_list() == [1, 2]
    assert result["prediction"].to_list() == [False, True]
    assert committee.models[0].received_data is data
    assert committee.models[1].received_data is data


@pytest.mark.parametrize("frame_factory", [make_dataframe, make_lazyframe])
def test_binned_committee_predict_forwards_all_layers_and_joins_results(
    frame_factory, isolated_executor
):
    future = isolated_executor.submit(
        binned_committee_predict_forwards_all_layers_and_joins_results,
        frame_factory=frame_factory,
    )
    future.result()


def binned_committee_predict_forwards_all_layers_and_joins_results(frame_factory):
    from neuralnet.models.binned_committee import BinnedCommittee

    class FakeModel:
        def __init__(self, ids: list[int], output: float):
            self.ids = ids
            self.output = output
            self.calls = []

        def predict(self, data, batch_size: int = 32, all_layers: bool = False):
            self.calls.append((data, batch_size, all_layers))
            result = pl.DataFrame(
                {
                    "id": self.ids,
                    "prediction": [self.output >= 0.5] * len(self.ids),
                    "output": [self.output] * len(self.ids),
                }
            )
            if all_layers:
                result = result.with_columns(
                    [
                        pl.Series("layer.dense.0", [1.0] * len(self.ids)),
                        pl.Series("layer.dense.1", [2.0] * len(self.ids)),
                    ]
                )
            if isinstance(data, pl.LazyFrame):
                return result.lazy()
            return result

    committee = BinnedCommittee([FakeModel([1], 0.1), FakeModel([4], 0.9)])

    data = frame_factory(
        {
            "id": [1, 2, 3, 4],
            "tag": [10, 20, 30, 40],
            "x": [0.5, 1.5, 0.5, 0.5],
            "y": [2.5, 2.5, 3.5, 2.5],
        }
    )

    result = collect_if_lazy(
        committee.predict_polars(data, all_layers=True, passthrough=True)
    )

    assert result.columns == [
        "id",
        "tag",
        "x",
        "y",
        "prediction",
        "output",
        "layer.dense.0",
        "layer.dense.1",
    ]
    assert result["id"].to_list() == [1, 2, 3, 4]
    assert result["tag"].to_list() == [10, 20, 30, 40]
    assert result["prediction"].to_list() == [False, None, None, True]
    assert result["output"].drop_nulls().to_list() == pytest.approx(
        [0.1, 0.9], rel=1e-5
    )
    assert result["layer.dense.0"].to_list() == [1.0, None, None, 1.0]
    assert result["layer.dense.1"].to_list() == [2.0, None, None, 2.0]
    assert committee.models[0].calls == [(data, 32, True)]
    assert committee.models[1].calls == [(data, 32, True)]

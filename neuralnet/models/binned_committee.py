from functools import cached_property
from typing import Any, Annotated, Self, overload, TYPE_CHECKING
import polars as pl
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
)
import numpy as np
import logging

from torch.nn import Sequential
from ..polars import PolarsFrame
from ..bins import VariableBin

if TYPE_CHECKING:
    from hgq.config import QuantizerConfig

type QuantizerConfigType = "QuantizerConfig" | None


def features_validator(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    elif isinstance(value, list):
        for i, val in enumerate(value):
            if not isinstance(val, str):
                raise TypeError(f"features[{i}] must be a string, got {type(val)}")
        return value
    else:
        raise TypeError(
            f"features must be a string or a list of strings, got {type(value)}"
        )


def validate_keras_sequential(value: Any) -> "Sequential":
    from keras import Sequential

    if isinstance(value, Sequential):
        return value
    else:
        raise TypeError(
            f"keras_model must be an instance of keras.Sequential, got {type(value)}"
        )


class BinnedModel:
    def __init__(
        self,
        bins: list[VariableBin],
        keras_model: "Sequential",
        features: list[str],
        decision_threshold: float = 0.5,
    ):
        self.bins = bins
        self.keras_model = keras_model
        self.features = features
        self.decision_threshold = decision_threshold

    @cached_property
    def input_cols(self) -> list[str]:
        return self.features + ["id"] + [bin.var_name for bin in self.bins]

    @cached_property
    def output_cols(self) -> list[str]:
        return ["id"] + self.features + ["prediction", "output"]

    def row_filter_expr(self) -> pl.Expr:
        expr = self.bins[0].as_polars_expr()
        for bin in self.bins[1:]:
            expr = expr & bin.as_polars_expr()
        return expr

    def predict_polars_batch(self, batch: pl.Series) -> pl.Series:
        data = np.stack(batch.to_numpy())
        prediction = self.predict_numpy(data)
        return pl.Series(prediction.flatten(), dtype=pl.Float32)

    def validate_schema(self, df: PolarsFrame) -> None:
        """
        Validates that the DataFrame has the required columns for prediction.
        Raises a ValueError if any required column is missing.
        """
        if isinstance(df, pl.LazyFrame):
            schema = df.collect_schema()
        else:
            schema = df.schema

        # validate id
        if "id" not in schema:
            raise ValueError("DataFrame must contain an 'id' column for prediction.")

        features_exceptions = []
        for feature in self.features:
            if feature not in schema:
                exc = TypeError(
                    f"Feature '{feature}' is missing from the DataFrame. Required features: {self.features}"
                )
                features_exceptions.append(exc)
                continue
            if not schema[feature].is_float():
                exc = TypeError(
                    f"Feature '{feature}' must be of type float, but got {schema[feature]}"
                )
                features_exceptions.append(exc)

        if features_exceptions:
            raise ExceptionGroup(
                "Errors in DataFrame schema features validation", features_exceptions
            )

    def predict_numpy(
        self, data: np.ndarray, batch_size: int = 32
    ) -> tuple[np.ndarray, np.ndarray[tuple[int,], np.bool_]]:
        output = self.keras_model.predict(data, batch_size=batch_size).flatten()
        prediction = np.where(output >= self.decision_threshold, True, False).astype(
            bool
        )
        return output, prediction

    def all_layers_predict_numpy(
        self, data: np.ndarray, batch_size: int = 32
    ) -> tuple[list[np.ndarray[np.floating]], list[str]]:

        from keras import Model

        model_input = self.keras_model.layers[0].input
        model_outputs = []
        output_names = []
        for layer in self.keras_model.layers:
            model_outputs.append(layer.output)
            output_names.append(layer.name)

        every_layer_model = Model(inputs=model_input, outputs=model_outputs)
        layer_predictions = every_layer_model.predict(data, batch_size=batch_size)
        if not isinstance(layer_predictions, list):
            layer_predictions = [layer_predictions]
        return layer_predictions, output_names

    def get_empty_output(self, is_lazy: bool) -> pl.DataFrame | pl.LazyFrame:
        """
        Returns an empty DataFrame with the same schema as the output of the model's predict method.
        """
        schema = {
            "id": pl.Int64,
            **{feature: pl.Float32 for feature in self.features},
            "prediction": pl.Boolean,
            "output": pl.Float32,
        }
        if is_lazy:
            return pl.LazyFrame(schema=schema)
        else:
            return pl.DataFrame(schema=schema)

    def predict_polars(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        batch_size: int = 32,
        all_layers: bool = False,
        passthrough: bool | list[str] = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        logger = logging.getLogger()
        selection = [pl.col("id")] + [pl.col(feature) for feature in self.features]
        filtered = data.filter(self.row_filter_expr()).select(*selection)

        self.validate_schema(filtered)
        input_is_lazy = isinstance(filtered, pl.LazyFrame)

        if input_is_lazy:
            filtered = filtered.collect()

        if filtered.is_empty():
            logger.warning(
                f"No data points found for the given bins ({self.bins}) and features ({self.features}.) Returning empty DataFrame."
            )
            return self.get_empty_output(input_is_lazy)

        features = filtered.select(pl.exclude("id")).to_numpy()
        result = filtered.drop(pl.exclude("id"))

        output, prediction = self.predict_numpy(features, batch_size=batch_size)
        result = result.with_columns(
            pl.Series(prediction.astype(np.bool_)).alias("prediction"),
        )
        del prediction

        if (len(output.shape) > 1) and output.shape[1] > 1:
            for i in range(output.shape[1]):
                result = result.with_columns(
                    pl.Series(output[:, i]).alias(f"output.{i}")
                )
        else:
            result = result.with_columns(pl.Series(output).alias("output"))
        del output

        if all_layers:
            layer_outputs, layer_names = self.all_layers_predict_numpy(
                features, batch_size
            )
            del features
            for layer_output, layer_name in zip(layer_outputs, layer_names):
                for output_idx in range(layer_output.shape[-1]):
                    result = result.with_columns(
                        pl.Series(layer_output[:, output_idx]).alias(
                            f"layer.{layer_name}.{output_idx}"
                        )
                    )

        if input_is_lazy:
            result = result.lazy()

        if passthrough:
            result = data.join(result, on="id", how="left")

        return result

    @overload
    def predict(
        self,
        data: pl.DataFrame,
        batch_size: int = 32,
        all_layers: bool = False,
        passthrough: bool = False,
    ) -> pl.DataFrame: ...

    @overload
    def predict(
        self,
        data: pl.LazyFrame,
        batch_size: int = 32,
        all_layers: bool = False,
        passthrough: bool = False,
    ) -> pl.LazyFrame: ...

    def predict(
        self,
        data,
        batch_size: int = 32,
        all_layers: bool = False,
        passthrough: bool = False,
    ):
        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            return self.predict_polars(
                data,
                batch_size=batch_size,
                all_layers=all_layers,
                passthrough=passthrough,
            )
        else:
            raise TypeError(
                f"Unsupported data type for prediction: {type(data)}. Expected polars DataFrame, polars LazyFrame, or numpy ndarray."
            )

    def quantize(
        self, kq_conf: QuantizerConfigType = None, bq_conf: QuantizerConfigType = None
    ) -> Self:
        from ..quantization.hgq import hgq_quantize

        quantized_keras_model = hgq_quantize(
            self.keras_model, kq_conf=kq_conf, bq_conf=bq_conf
        )

        quantized_binned_model = BinnedModel(
            bins=self.bins,
            keras_model=quantized_keras_model,
            features=self.features,
            decision_threshold=self.decision_threshold,
        )
        return quantized_binned_model


class BinnedCommittee:
    
    def __init__(self, models: list[BinnedModel]):
        if len(models) < 1:
            raise ValueError(
                "BinnedCommittee must contain at least two BinnedModel instances."
            )
        self.models = models

    @cached_property
    def input_cols(self) -> list[str]:
        all_inputs = set()
        for model in self.models:
            all_inputs.update(model.input_cols)
        return list(all_inputs)

    @cached_property
    def output_cols(self) -> list[str]:
        all_outputs = set()
        for model in self.models:
            all_outputs.update(model.output_cols)
        return list(all_outputs)

    @overload
    def predict_polars(
        self, data: pl.LazyFrame, batch_size: int = 32
    ) -> pl.DataFrame: ...

    @overload
    def predict_polars(
        self, data: pl.DataFrame, batch_size: int = 32
    ) -> pl.DataFrame: ...

    def predict_polars(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        batch_size: int = 32,
        all_layers: bool = False,
        passthrough: bool = False,
    ) -> pl.DataFrame:
        prediction_df = []
        for model in self.models:
            prediction = model.predict(
                data, batch_size=batch_size, all_layers=all_layers
            )
            prediction_df.append(prediction)
        results = pl.concat(prediction_df)
        del prediction_df
        if passthrough:
            results = data.join(results, on="id", how="left")
        return results

    @overload
    def predict(self, data: pl.DataFrame, batch_size: int = 32) -> pl.DataFrame: ...

    @overload
    def predict(self, data: pl.LazyFrame, batch_size: int = 32) -> pl.DataFrame: ...

    def predict(self, data, batch_size: int = 32):
        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            return self.predict_polars(data, batch_size=batch_size)
        else:
            raise TypeError(
                f"Unsupported data type for prediction: {type(data)}. Expected polars DataFrame or polars LazyFrame."
            )

    def quantize(
        self, kq_conf: QuantizerConfigType = None, bq_conf: QuantizerConfigType = None
    ) -> Self:
        quantized_models = [model.quantize(kq_conf, bq_conf) for model in self.models]
        return self.model_copy(update={"models": quantized_models})

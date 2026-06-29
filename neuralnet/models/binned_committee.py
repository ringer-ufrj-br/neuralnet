from typing import Any, Annotated, Self, Literal, overload
import polars as pl
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TYPE_CHECKING
import numpy as np
import logging
from functools import cached_property

from torch.nn import Sequential
from ..polars import PolarsFrame

if TYPE_CHECKING:
    from hgq.config import QuantizerConfig

type QuantizerConfigType = "QuantizerConfig" | None


class VariableBin(BaseModel):
    var_name: Annotated[
        str, Field(description="Name of the variable used in the binning.")
    ]
    lower: Annotated[float, Field(description="Lower bound of the bin.")]
    upper: Annotated[float, Field(description="Upper bound of the bin.")]
    closed: Annotated[
        Literal["left", "right", "both", "none"],
        Field(
            description="Indicates whether the bin is closed on the left, right, both, or neither side."
        ),
    ] = "left"

    def is_inside_numpy(self, value):
        if self.closed == "left":
            return self.lower <= value < self.upper
        elif self.closed == "right":
            return self.lower < value <= self.upper
        elif self.closed == "both":
            return self.lower <= value <= self.upper
        elif self.closed == "none":
            return self.lower < value < self.upper
        else:
            raise ValueError(f"Invalid closed value: {self.closed}")

    def as_polars_expr(self) -> pl.Expr:
        return pl.col(self.var_name).is_between(
            self.lower, self.upper, closed=self.closed
        )

    @overload
    def apply_bin(self, df: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def apply_bin(self, df: pl.LazyFrame) -> pl.LazyFrame: ...

    def is_inside_polars(self, df: PolarsFrame) -> PolarsFrame:
        return df.filter(self.as_polars_expr())


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


class BinnedModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bins: Annotated[
        list[VariableBin],
        Field(
            description="List of VariableBin instances defining the binning criteria for the model."
        ),
    ]
    keras_model: Annotated[
        Any,
        Field(description="The Keras model instance for the binned model."),
        BeforeValidator(validate_keras_sequential),
    ]
    features: Annotated[
        list[str],
        Field(description="List of feature names to be used in the model."),
        BeforeValidator(features_validator),
    ]
    decision_threshold: Annotated[
        float,
        Field(description="Decision threshold for the binary classification model."),
    ]
    fold: int | None = None
    fold_col: Annotated[
        str | None,
        Field(
            description="Name of the column in the DataFrame that indicates the fold for cross-validation. If None, no fold filtering is applied."
        ),
    ] = None

    def row_filter_expr(self) -> pl.Expr:
        expr = self.bins[0].as_polars_expr()
        for bin in self.bins[1:]:
            expr = expr & bin.as_polars_expr()
        return expr

    def predict_polars_batch(self, batch: pl.Series) -> pl.Series:
        data = np.stack(batch.to_numpy())
        prediction = self.predict_numpy(data)
        return pl.Series(prediction.flatten(), dtype=pl.Float32)

    @cached_property
    def fold_col_expr(self) -> pl.Expr | None:
        if self.fold_col is None or self.fold is None:
            return None
        return (
            pl.when(self.fold_col == self.fold)
            .then(True)
            .when(self.fold_col.is_null())
            .then(None)
            .otherwise(False)
            .alias("is_val_dataset")
        )

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

        for feature in self.features:
            if feature not in schema:
                raise TypeError(
                    f"Feature '{feature}' is missing from the DataFrame. Required features: {self.features}"
                )
            if not schema[feature].is_float():
                raise TypeError(
                    f"Feature '{feature}' must be of type float, but got {schema[feature]}"
                )

        if self.fold_col is None:
            return

        if self.fold_col not in schema:
            raise TypeError(
                f"Fold column '{self.fold_col}' is missing from the DataFrame. Required fold column: {self.fold_col}"
            )
        if not (schema[self.fold_col].is_integer() or schema[self.fold_col].is_float()):
            raise TypeError(
                f"Fold column '{self.fold_col}' must be of type int, but got {schema[self.fold_col]}"
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

    def predict_polars(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        batch_size: int = 32,
        output_layers: bool = False,
    ) -> pl.DataFrame:
        logger = logging.getLogger()
        selection = [pl.col("id")] + [pl.col(feature) for feature in self.features]
        if self.fold_col_expr is not None:
            selection.append(self.fold_col_expr)
        filtered = data.filter(self.row_filter_expr()).select(*selection)

        self.validate_schema(filtered)

        if isinstance(filtered, pl.LazyFrame):
            filtered = filtered.collect()

        if filtered.is_empty():
            logger.warning(
                f"No data points found for the given bins ({self.bins}) and features ({self.features}.) Returning empty DataFrame."
            )
            filtered.clear()  # Frees memory premptively
            del filtered
            return pl.DataFrame(schema=filtered.schema)

        if self.fold_col_expr is not None:
            is_val_dataset = (
                filtered.select(pl.col("is_val_dataset")).to_numpy().flatten()
            )
            filtered = filtered.drop(pl.col("is_val_dataset"))

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

        if output_layers:
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
        else:
            del features  # Frees memory premptively

        if self.fold_col_expr is not None:
            result = result.with_columns(
                pl.Series(is_val_dataset).alias("is_val_dataset")
            )
        return result

    @overload
    def predict(self, data: pl.DataFrame, batch_size: int = 32) -> pl.DataFrame: ...

    @overload
    def predict(self, data: pl.LazyFrame, batch_size: int = 32) -> pl.DataFrame: ...

    def predict(self, data, batch_size: int = 32, output_layers: bool = False):
        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            return self.predict_polars(
                data, batch_size=batch_size, output_layers=output_layers
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    models: list[BinnedModel]

    @overload
    def predict_polars(
        self, data: pl.LazyFrame, batch_size: int = 32
    ) -> pl.DataFrame: ...

    @overload
    def predict_polars(
        self, data: pl.DataFrame, batch_size: int = 32
    ) -> pl.DataFrame: ...

    def predict_polars(
        self, data: pl.LazyFrame | pl.DataFrame, batch_size: int = 32
    ) -> pl.DataFrame:
        prediction_df = []
        for model in self.models:
            prediction = model.predict(data, batch_size=batch_size)
            prediction_df.append(prediction)
        return pl.concat(prediction_df)

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

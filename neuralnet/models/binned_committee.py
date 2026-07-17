"""Specialist committee models for bin-conditioned inference.

This module provides abstractions to route samples to specialist models based
on bin expressions over input variables.
"""

from functools import cached_property
from typing import Self, overload, TYPE_CHECKING, Literal
import polars as pl
import numpy as np
import logging

from torch.nn import Sequential
from ..bins import VariableBin

if TYPE_CHECKING:
    from ..quantization.hgq import HGQFixedPointConfig


type FixedPointQuantizationBackendType = Literal["hgq", "custom"]


class BinnedModel:
    """Specialist model bound to a bin selector.

    A ``BinnedModel`` applies one Keras model to the subset of rows selected by
    its bin expressions and returns predictions aligned by sample ``id``.

    Attributes
    ----------
    bins : list[VariableBin]
        Bin expressions defining which rows are handled by this specialist.
    keras_model : Sequential
        Keras model used for inference on selected rows.
    features : list[str]
        Feature columns consumed by the model.
    decision_threshold : float or None
        Optional threshold used to derive a boolean ``prediction`` column.
    """

    def __init__(
        self,
        bins: list[VariableBin],
        keras_model: "Sequential",
        features: list[str],
        decision_threshold: float | None = None,
    ):
        """Initialize a specialist model for one bin selection.

        Parameters
        ----------
        bins : list[VariableBin]
            Bin expressions defining which rows are handled by this specialist.
        keras_model : Sequential
            Keras model used for inference on selected rows.
        features : list[str]
            Feature columns consumed by the model.
        decision_threshold : float or None, default=None
            Optional threshold used to derive a boolean ``prediction`` column.
        """

        self.bins = bins
        if not isinstance(keras_model, Sequential):
            raise TypeError(
                f"keras_model must be an instance of keras.Sequential, got {type(keras_model)}"
            )
        self.keras_model = keras_model
        if isinstance(features, list):
            for i, val in enumerate(features):
                if not isinstance(val, str):
                    raise TypeError(f"features[{i}] must be a string, got {type(val)}")
        self.features = features
        self.decision_threshold = decision_threshold

    @cached_property
    def input_cols(self) -> list[str]:
        """Columns required for this specialist input.

        Returns
        -------
        list[str]
            Required columns composed of features, ``id``, and bin variables.
        """

        return self.features + ["id"] + [bin.var_name for bin in self.bins]

    @cached_property
    def output_cols(self) -> list[str]:
        """Columns produced by this specialist.

        Returns
        -------
        list[str]
            Output column names, including ``prediction`` when a decision
            threshold is configured.
        """

        if self.decision_threshold is not None:
            return ["id"] + self.features + ["prediction", "output"]
        return ["id"] + self.features + ["output"]

    def row_filter_expr(self) -> pl.Expr:
        """Build the Polars boolean expression selecting this specialist bin.

        Returns
        -------
        pl.Expr
            Conjunction of all bin expressions in ``self.bins``.
        """

        expr = self.bins[0].as_polars_expr()
        for bin in self.bins[1:]:
            expr = expr & bin.as_polars_expr()
        return expr

    def validate_schema(self, df: pl.LazyFrame | pl.DataFrame) -> None:
        """Validate input schema required by this specialist.

        Parameters
        ----------
        df : pl.LazyFrame or pl.DataFrame
            Input table to be validated.

        Raises
        ------
        ValueError
            If ``id`` is missing from the input schema.
        ExceptionGroup
            If one or more feature columns are missing or not float-like.
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

    def all_layers_predict_numpy(
        self, data: np.ndarray, batch_size: int = 32
    ) -> tuple[list[np.ndarray[np.floating]], list[str]]:
        """Run prediction and return outputs for every layer.

        Parameters
        ----------
        data : np.ndarray
            Model input array.
        batch_size : int, default=32
            Batch size used during inference.

        Returns
        -------
        tuple[list[np.ndarray[np.floating]], list[str]]
            Layer outputs and their corresponding layer names in model order.
        """

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
        """Create an empty prediction table with the expected output schema.

        Parameters
        ----------
        is_lazy : bool
            Whether to return a lazy output table.

        Returns
        -------
        pl.DataFrame or pl.LazyFrame
            Empty prediction table using this specialist output schema.
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
        """Predict using Polars input and optional specialist diagnostics.

        Parameters
        ----------
        data : pl.DataFrame or pl.LazyFrame
            Input table containing ``id`` and feature columns.
        batch_size : int, default=32
            Batch size used during Keras inference.
        all_layers : bool, default=False
            Whether to append per-layer outputs.
        passthrough : bool or list[str], default=False
            Whether to left-join predictions back to the original input.

        Returns
        -------
        pl.DataFrame or pl.LazyFrame
            Prediction table for rows selected by this specialist.
        """

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

        output = self.keras_model.predict(features, batch_size=batch_size).flatten()
        if self.decision_threshold is not None:
            prediction = np.where(
                output >= self.decision_threshold, True, False
            ).astype(bool)
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
        """Dispatch prediction for supported input table types.

        Parameters
        ----------
        data : Any
            Input data. Supported types are ``pl.DataFrame`` and
            ``pl.LazyFrame``.
        batch_size : int, default=32
            Batch size used during Keras inference.
        all_layers : bool, default=False
            Whether to append per-layer outputs.
        passthrough : bool, default=False
            Whether to left-join predictions back to the original input.

        Returns
        -------
        pl.DataFrame or pl.LazyFrame
            Prediction table generated by ``predict_polars``.

        Raises
        ------
        TypeError
            If ``data`` is not a supported Polars table type.
        """

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

    def fixed_point_quantization(
        self,
        weight_config: "HGQFixedPointConfig" = None,
        bias_config: "HGQFixedPointConfig" = None,
        backend: FixedPointQuantizationBackendType = "hgq",
    ) -> Self:
        """Create a quantized copy of this specialist model.

        Parameters
        ----------
        weight_config : HGQFixedPointConfig or None, default=None
            Weight quantization configuration.
        bias_config : HGQFixedPointConfig or None, default=None
            Bias quantization configuration.
        backend : {"hgq", "custom"}, default="hgq"
            Quantization backend implementation.

        Returns
        -------
        Self
            New ``BinnedModel`` instance with a quantized Keras model.
        """

        if backend == "hgq":
            from ..quantization.hgq import hgq_quantize

            quantized_keras_model = hgq_quantize(
                self.keras_model,
                weight_quantizer_config=weight_config,
                bias_quantizer_config=bias_config,
            )
        elif backend == "custom":
            from ..quantization.keras import fixed_point_quantize
            from .. import get_logger

            logger = get_logger()
            logger.warning(
                "Using custom fixed-point quantization."
                "This is a simplified implementation and may not be as accurate as the HGQ backend. "
                "Consider using the HGQ backend for better quantization results. "
                "Custom fixed-point quantization does not work with bias quantization. "
                "If you need bias quantization, please use the HGQ backend."
            )
            quantized_keras_model = fixed_point_quantize(
                self.keras_model,
                floating_bits=weight_config.f0,
                integer_bits=weight_config.i0,
            )

        quantized_binned_model = BinnedModel(
            bins=self.bins,
            keras_model=quantized_keras_model,
            features=self.features,
            decision_threshold=self.decision_threshold,
        )
        return quantized_binned_model


class BinnedCommittee:
    """Collection of specialist ``BinnedModel`` instances.

    The committee evaluates each specialist and concatenates their outputs to
    form the final prediction table.

    Attributes
    ----------
    models : list[BinnedModel]
        Specialist models included in the committee.
    """

    def __init__(self, models: list[BinnedModel]):
        """Initialize a committee of specialist binned models.

        Parameters
        ----------
        models : list[BinnedModel]
            Specialist models to include in the committee.

        Raises
        ------
        ValueError
            If fewer than one specialist model is provided.
        """

        if len(models) < 1:
            raise ValueError(
                "BinnedCommittee must contain at least two BinnedModel instances."
            )
        self.models = models

    @cached_property
    def input_cols(self) -> list[str]:
        """Union of all input columns required by committee specialists.

        Returns
        -------
        list[str]
            Unique input column names across all specialist models.
        """

        all_inputs = set()
        for model in self.models:
            all_inputs.update(model.input_cols)
        return list(all_inputs)

    @cached_property
    def output_cols(self) -> list[str]:
        """Union of all output columns emitted by committee specialists.

        Returns
        -------
        list[str]
            Unique output column names across all specialist models.
        """

        all_outputs = set()
        for model in self.models:
            all_outputs.update(model.output_cols)
        return list(all_outputs)

    @overload
    def predict_polars(
        self,
        data: pl.LazyFrame,
        batch_size: int = 32,
        all_layers: bool = False,
        passthrough: bool = False,
    ) -> pl.DataFrame: ...

    @overload
    def predict_polars(
        self,
        data: pl.DataFrame,
        batch_size: int = 32,
        all_layers: bool = False,
        passthrough: bool = False,
    ) -> pl.DataFrame: ...

    def predict_polars(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        batch_size: int = 32,
        all_layers: bool = False,
        passthrough: bool = False,
    ) -> pl.DataFrame:
        """Run prediction for all specialists and concatenate results.

        Parameters
        ----------
        data : pl.LazyFrame or pl.DataFrame
            Input table passed to each specialist model.
        batch_size : int, default=32
            Batch size used during model inference.
        all_layers : bool, default=False
            Whether to include per-layer outputs from each specialist.
        passthrough : bool, default=False
            Whether to left-join predictions back to the input table.

        Returns
        -------
        pl.DataFrame
            Concatenated prediction table from all committee specialists.
        """

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
        """Dispatch committee prediction for supported Polars table types.

        Parameters
        ----------
        data : Any
            Input data. Supported types are ``pl.DataFrame`` and
            ``pl.LazyFrame``.
        batch_size : int, default=32
            Batch size used during model inference.

        Returns
        -------
        pl.DataFrame
            Committee prediction table.

        Raises
        ------
        TypeError
            If ``data`` is not a supported Polars table type.
        """

        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            return self.predict_polars(data, batch_size=batch_size)
        else:
            raise TypeError(
                f"Unsupported data type for prediction: {type(data)}. Expected polars DataFrame or polars LazyFrame."
            )

    def fixed_point_quantization(
        self,
        weight_config: "HGQFixedPointConfig" = None,
        bias_config: "HGQFixedPointConfig" = None,
        backend: FixedPointQuantizationBackendType = "hgq",
    ) -> Self:
        """Create a quantized copy of every specialist in the committee.

        Parameters
        ----------
        weight_config : HGQFixedPointConfig or None, default=None
            Weight quantization configuration.
        bias_config : HGQFixedPointConfig or None, default=None
            Bias quantization configuration.
        backend : {"hgq", "custom"}, default="hgq"
            Quantization backend implementation.

        Returns
        -------
        Self
            New ``BinnedCommittee`` containing quantized specialist models.
        """

        quantized_models = [
            model.fixed_point_quantization(weight_config, bias_config, backend=backend)
            for model in self.models
        ]
        return BinnedCommittee(models=quantized_models)

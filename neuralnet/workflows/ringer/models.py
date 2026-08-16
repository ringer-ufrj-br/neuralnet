from functools import cached_property
from typing import TYPE_CHECKING, Protocol, Self, overload, runtime_checkable
import logging

import numpy as np
import polars as pl

from ...bins import VariableBin
from ...quantization.quantizers import FixedPointQuantizer

if TYPE_CHECKING:
    from keras import Sequential


@runtime_checkable
class Tranformation(Protocol):
    """Contract for preprocessing objects used by specialist inference."""

    input_cols: list[str]
    output_cols: list[str]

    def __call__(self, data: pl.DataFrame | pl.LazyFrame, passthrough: bool = False) -> pl.DataFrame | pl.LazyFrame:
        """Apply processing to the input table.

        Parameters
        ----------
        data : pl.DataFrame or pl.LazyFrame
            Input table to transform.
        passthrough : bool, default=False
            Whether to preserve non-feature columns in the transformed output.

        Returns
        -------
        pl.DataFrame or pl.LazyFrame
            Transformed table.
        """

        ...


class BinnedSpecialistModel:
    """Complete inference pipeline for a single specialist model.

    The specialist filters rows by the configured bin expressions, applies the
    preprocessing pipeline, and then runs the Keras model on the preprocessed
    feature columns.
    """

    def __init__(
        self,
        bins: list[VariableBin],
        keras_model: "Sequential",
        preprocessing: Tranformation,
        decision_threshold: float | None = None,
        fold: int | None = None,
        training_results: dict[str, object] | None = None,
    ):
        """Initialize a specialist inference pipeline.

        Parameters
        ----------
        bins : list[VariableBin]
            Bin expressions defining which rows are handled by this specialist.
        keras_model : Sequential
            Keras model used for inference on selected rows.
        preprocessing : callable
            Preprocessing pipeline applied after bin filtering.
        decision_threshold : float or None, default=None
            Optional threshold used to derive a boolean ``prediction`` column.
        training_results : dict[str, object] or None, default=None
            Optional training results associated with this specialist model.
        """

        from keras import Sequential

        self.bins = bins
        if not isinstance(keras_model, Sequential):
            raise TypeError(f"keras_model must be an instance of keras.Sequential, got {type(keras_model)}")
        self.keras_model = keras_model
        if not isinstance(preprocessing, Tranformation):
            raise TypeError(
                "preprocessing must implement the Tranformation protocol with input_cols, output_cols, and __call__"
            )
        self.preprocessing = preprocessing
        self.model_input_cols = list(preprocessing.output_cols)

        self.decision_threshold = decision_threshold

        model_input_shape = getattr(self.keras_model, "input_shape", None)
        if isinstance(model_input_shape, tuple) and len(model_input_shape) > 1:
            expected_width = model_input_shape[-1]
            if expected_width is not None and expected_width != len(self.model_input_cols):
                raise ValueError(
                    "keras_model input width must match preprocessing.output_cols: "
                    f"expected {len(self.model_input_cols)}, got {expected_width}"
                )

        self.fold = fold
        self.training_results = training_results

    @cached_property
    def input_cols(self) -> set[str]:
        """Columns required by this specialist before preprocessing.

        Returns
        -------
        set[str]
            Required input columns composed of ``id``, bin variables, and
            preprocessing input columns.
        """
        return {"id"} | {bin.var_name for bin in self.bins} | set(self.preprocessing.input_cols)

    @cached_property
    def output_cols(self) -> set[str]:
        """Columns produced by this specialist.

        Returns
        -------
        set[str]
            Output columns emitted by prediction, including ``prediction`` when
            ``decision_threshold`` is configured.
        """

        output_cols = {"id", "output"}
        if self.decision_threshold is not None:
            output_cols |= {"prediction"}
        return output_cols

    def row_filter_expr(self) -> pl.Expr:
        """Build the Polars boolean expression selecting this specialist bin."""

        expr = pl.all_horizontal(*[bin.as_polars_expr() for bin in self.bins])
        return expr

    def validate_schema(self, df: pl.LazyFrame | pl.DataFrame) -> None:
        """Validate the input schema required by this specialist."""

        if isinstance(df, pl.LazyFrame):
            schema = df.collect_schema()
        else:
            schema = df.schema

        missing_columns = [col for col in self.input_cols if col not in schema]
        if missing_columns:
            raise ValueError(
                f"DataFrame must contain the following columns for specialist inference: {missing_columns}"
            )

    def get_empty_output(self, is_lazy: bool) -> pl.DataFrame | pl.LazyFrame:
        """Create an empty prediction table with the expected output schema."""

        schema = {
            "id": pl.Int64,
            **{feature: pl.Float32 for feature in self.model_input_cols},
        }
        if self.decision_threshold is not None:
            schema["prediction"] = pl.Boolean
        schema["output"] = pl.Float32

        if is_lazy:
            return pl.LazyFrame(schema=schema)
        return pl.DataFrame(schema=schema)

    def predict_polars(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        batch_size: int = 32,
        passthrough: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Predict using Polars input and the specialist preprocessing pipeline."""

        logger = logging.getLogger()
        self.validate_schema(data)

        input_is_lazy = isinstance(data, pl.LazyFrame)
        filtered = (
            data.filter(self.row_filter_expr())
            .pipe(self.preprocessing, passthrough=True)
            .select([pl.col("id")] + [pl.col(feature) for feature in self.model_input_cols])
        )

        if isinstance(filtered, pl.LazyFrame):
            filtered = filtered.collect()

        if filtered.is_empty():
            logger.warning(f"No data points found for the given bins ({self.bins}). Returning empty DataFrame.")
            return self.get_empty_output(input_is_lazy)

        features = filtered.drop("id").to_numpy()
        result = filtered.select("id")

        output = self.keras_model.predict(features, batch_size=batch_size)
        output_array = np.asarray(output)

        if self.decision_threshold is not None:
            if output_array.ndim > 1 and output_array.shape[1] > 1:
                threshold_input = output_array[:, 0]
            else:
                threshold_input = output_array.reshape(-1)
            prediction = (threshold_input >= self.decision_threshold).astype(bool)
            result = result.with_columns(
                pl.Series(prediction.astype(np.bool_)).alias("prediction"),
            )

        if output_array.ndim > 1 and output_array.shape[1] > 1:
            for i in range(output_array.shape[1]):
                result = result.with_columns(pl.Series(output_array[:, i]).alias(f"output.{i}"))
        else:
            result = result.with_columns(pl.Series(output_array.reshape(-1)).alias("output"))

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
        passthrough: bool = False,
    ) -> pl.DataFrame: ...

    @overload
    def predict(
        self,
        data: pl.LazyFrame,
        batch_size: int = 32,
        passthrough: bool = False,
    ) -> pl.LazyFrame: ...

    def predict(
        self,
        data,
        batch_size: int = 32,
        passthrough: bool = False,
    ):
        """Dispatch prediction for supported Polars table types."""

        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            return self.predict_polars(
                data,
                batch_size=batch_size,
                passthrough=passthrough,
            )
        raise TypeError(
            f"Unsupported data type for prediction: {type(data)}. Expected polars DataFrame or polars LazyFrame."
        )

    def fixed_point_quantization(
        self, norm_quantizer: "FixedPointQuantizer", nn_quantizer: "FixedPointQuantizer"
    ) -> Self:
        """Create a quantized copy of this specialist inference model.

        Parameters
        ----------
        norm_quantizer : FixedPointQuantizer
            Quantizer applied to the preprocessing pipeline.
        nn_quantizer : FixedPointQuantizer
            Quantizer applied to the Keras model weights.

        Returns
        -------
        Self
            New ``BinnedSpecialistModel`` instance with a quantized Keras model and preprocessing.
        """

        from ...quantization.keras import fixed_point_quantize

        if isinstance(nn_quantizer, dict):
            nn_quantizer = FixedPointQuantizer(**nn_quantizer)

        if isinstance(norm_quantizer, dict):
            norm_quantizer = FixedPointQuantizer(**norm_quantizer)

        quantized_keras_model = fixed_point_quantize(
            self.keras_model,
            floating_bits=nn_quantizer.fractional_bits,
            integer_bits=nn_quantizer.integer_bits,
        )

        quantized_preprocessing = self.preprocessing.fixed_point_quantization(norm_quantizer)

        return BinnedSpecialistModel(
            bins=self.bins,
            keras_model=quantized_keras_model,
            preprocessing=quantized_preprocessing,
            decision_threshold=self.decision_threshold,
            fold=self.fold,
            training_results=self.training_results,
        )


class BinnedSpecialistCommittee:
    """Collection of specialist models for bin-conditioned inference."""

    def __init__(self, models: list[BinnedSpecialistModel]):
        """Initialize a specialist committee.

        Parameters
        ----------
        models : list[BinnedSpecialistModel]
            Specialist models to include in the committee.

        Raises
        ------
        TypeError
            If ``models`` is not a list of ``BinnedSpecialistModel`` objects.
        ValueError
            If no models are provided or if model input/output schemas differ.
        """

        if not isinstance(models, list):
            raise TypeError(f"models must be a list of BinnedSpecialistModel, got {type(models)}")
        if len(models) < 1:
            raise ValueError("BinnedSpecialistCommittee must contain at least one BinnedSpecialistModel instance.")

        for idx, model in enumerate(models):
            if not isinstance(model, BinnedSpecialistModel):
                raise TypeError(f"models[{idx}] must be a BinnedSpecialistModel, got {type(model)}")

        reference_input_cols = models[0].input_cols
        reference_output_cols = models[0].output_cols
        for idx, model in enumerate(models[1:], start=1):
            if model.input_cols != reference_input_cols:
                raise ValueError(
                    "All specialist models must have identical input_cols for committee concatenation. "
                    f"models[0]={reference_input_cols}, models[{idx}]={model.input_cols}"
                )
            if model.output_cols != reference_output_cols:
                raise ValueError(
                    "All specialist models must have identical output_cols for committee concatenation. "
                    f"models[0]={reference_output_cols}, models[{idx}]={model.output_cols}"
                )

        self.models = models

    @cached_property
    def input_cols(self) -> set[str]:
        """Input columns expected by committee specialists.

        Returns
        -------
        set[str]
            Union of required input columns across all specialists.
        """

        all_inputs: set[str] = set()
        for model in self.models:
            all_inputs.update(model.input_cols)
        return all_inputs

    @cached_property
    def output_cols(self) -> set[str]:
        """Output columns produced by committee specialists.

        Returns
        -------
        set[str]
            Union of output columns across all specialists.
        """

        all_outputs: set[str] = set()
        for model in self.models:
            all_outputs.update(model.output_cols)
        return all_outputs

    @overload
    def predict(
        self,
        data: pl.DataFrame,
        batch_size: int = 32,
        passthrough: bool = False,
    ) -> pl.DataFrame: ...

    @overload
    def predict(
        self,
        data: pl.LazyFrame,
        batch_size: int = 32,
        passthrough: bool = False,
    ) -> pl.LazyFrame: ...

    def predict(self, data, batch_size: int = 32, passthrough: bool = False, with_specialist_id: bool = False):
        """Run inference for every specialist and concatenate outputs.

        Parameters
        ----------
        data : pl.DataFrame or pl.LazyFrame
            Input table passed to each specialist model.
        batch_size : int, default=32
            Batch size used by each specialist model during Keras inference.
        passthrough : bool, default=False
            Whether to left-join committee predictions back to the input table.
        with_specialist_id : bool, default=False
            Whether to include a ``specialist_id`` column in the output indicating
            which specialist produced each prediction.

        Returns
        -------
        pl.DataFrame or pl.LazyFrame
            Concatenated committee predictions.

        Raises
        ------
        TypeError
            If ``data`` is not a Polars DataFrame or LazyFrame.
        """

        if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError(
                f"Unsupported data type for prediction: {type(data)}. Expected polars DataFrame or polars LazyFrame."
            )

        predictions = []
        for model in self.models:
            model_predictions = model.predict(data, batch_size=batch_size, passthrough=False)
            if with_specialist_id:
                model_predictions = model_predictions.with_columns(
                    pl.lit(model.training_results["id"], dtype=pl.UInt32).alias("specialist_id")
                )
            predictions.append(model_predictions)

        results = pl.concat(predictions)

        if passthrough:
            results = data.join(results, on="id", how="left")
        return results

    def fixed_point_quantization(
        self, norm_quantizer: "FixedPointQuantizer", nn_quantizer: "FixedPointQuantizer"
    ) -> Self:
        """Create a quantized copy of every specialist in the committee.

        Parameters
        ----------
        norm_quantizer : FixedPointQuantizer
            Quantizer applied to the preprocessing pipeline of each specialist.
        nn_quantizer : FixedPointQuantizer
            Quantizer applied to the Keras model weights of each specialist.

        Returns
        -------
        Self
            New ``BinnedSpecialistCommittee`` containing quantized specialist models.
        """

        quantized_models = [
            model.fixed_point_quantization(norm_quantizer=norm_quantizer, nn_quantizer=nn_quantizer)
            for model in self.models
        ]
        return BinnedSpecialistCommittee(models=quantized_models)

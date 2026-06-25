import polars as pl
import numpy as np
from typing import Annotated, overload
from pydantic import Field, AfterValidator, BaseModel, ConfigDict
from functools import cached_property
from .quantization import FixedPointQuantizer
from .utils import get_ring_slices_per_layer

POLARS_TO_NUMPY_DTYPE = {
    pl.Int8: np.int8,
    pl.Int16: np.int16,
    pl.Int32: np.int32,
    pl.Int64: np.int64,
    pl.UInt8: np.uint8,
    pl.UInt16: np.uint16,
    pl.UInt32: np.uint32,
    pl.UInt64: np.uint64,
    pl.Float32: np.float32,
    pl.Float64: np.float64,
    pl.Boolean: np.bool_,
}


def polars_expression_validator(
    value: pl.Expr | str,
) -> pl.Expr:
    if isinstance(value, pl.Expr):
        return value
    elif isinstance(value, str):
        return pl.col(value)
    else:
        raise ValueError(f"Invalid Polars expression: {value}")


type PolarsExpression = Annotated[
    pl.Expr | str,
    Field(
        description="A Polars expression, which can be a column name or sql expression (str) or a Polars expression (pl.Expr).",
        example="column_name",
    ),
    AfterValidator(polars_expression_validator),
]

type PolarsFrame = pl.DataFrame | pl.LazyFrame


class AlternativeNorm1(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_col: PolarsExpression
    output_col: Annotated[str | None, Field(default=None)]

    def model_post_init(self, __context) -> None:
        if self.output_col is None:
            self.output_col = f"{self.input_col.meta.output_name()}_alternative_norm1"

    @cached_property
    def list_polars_expr(self) -> pl.Expr:
        norms = self.input_col.list.sum()
        norms = pl.when(norms == 0).then(1).otherwise(norms)
        return (self.input_col / norms).alias(self.output_col)
    
    @cached_property
    def arr_polars_expr(self) -> pl.Expr:
        norms = self.input_col.arr.sum()
        norms = pl.when(norms == 0).then(1).otherwise(norms)
        return (self.input_col / norms).alias(self.output_col)
    
    def get_expr(self, data: pl.DataFrame | pl.LazyFrame) -> pl.Expr:
        if isinstance(data, pl.DataFrame):
            schema = data.schema
        elif isinstance(data, pl.LazyFrame):
            schema = data.collect_schema()
        else:
            raise TypeError(f"Expected pl.DataFrame or pl.LazyFrame, got {type(data)}")
        
        col_dtype = schema[self.input_col.meta.output_name()]
        if isinstance(col_dtype, pl.List):
            return self.list_polars_expr
        elif isinstance(col_dtype, pl.Array):
            return self.arr_polars_expr
        else:
            raise TypeError(f"Expected list or array column, got {col_dtype}")

    @overload
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def __call__(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def __call__(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        expr = self.get_expr(data)
        return data.with_columns(expr)

    def fixed_point_quantization(
        self, quantizer: FixedPointQuantizer
    ) -> "FixedPointQuantizedAlternativeNorm1":
        return FixedPointQuantizedAlternativeNorm1(
            input_col=self.input_col, output_col=self.output_col, quantizer=quantizer
        )


class FixedPointQuantizedAlternativeNorm1(AlternativeNorm1):
    quantizer: FixedPointQuantizer = Field(
        description="Fixed-point quantizer configuration."
    )

    @cached_property
    def list_polars_expr(self) -> pl.Expr:
        return super().list_polars_expr.list.eval(
            pl.element().pipe(self.quantizer.quantize_polars_expr)
        )

    @cached_property
    def arr_polars_expr(self) -> pl.Expr:
        return super().arr_polars_expr.arr.eval(
            pl.element().pipe(self.quantizer.quantize_polars_expr)
        )

    @overload
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def __call__(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def __call__(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        expr = self.get_expr(data)
        return data.with_columns(expr)


class RingSlicesPerLayer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    fraction: Annotated[int, Field(gt=0, description="Fraction of rings to select per layer.")]
    to_array: Annotated[
        bool,
        Field(default=False, description="Whether to convert the list to an array after gathering.")
    ]

    input_col: PolarsExpression
    output_col: Annotated[str | None, Field(default=None)]

    def model_post_init(self, __context) -> None:
        if self.output_col is None:
            self.output_col = f"{self.input_col.meta.output_name()}_slices_per_layer"

    @cached_property
    def polars_expr(self) -> pl.Expr:
        idxs = get_ring_slices_per_layer(self.fraction)
        result = self.input_col.list.gather(idxs).alias(self.output_col)
        if self.to_array:
            result = result.list.to_array(len(idxs))
        return result

    @overload
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def __call__(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def __call__(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        return data.with_columns(self.polars_expr)

import polars as pl
import numpy as np
from typing import Annotated, overload
from pydantic import Field, AfterValidator, BaseModel
from functools import cached_property
from .quantization import FixedPointQuantizer

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
    input_col: PolarsExpression
    output_col: Annotated[str | None, Field(default=None)]

    def model_post_init(self, __context) -> None:
        if self.output_col is None:
            self.output_col = f"{self.input_col.meta.output_name()}_alternative_norm1"

    @cached_property
    def polars_expr(self) -> pl.Expr:
        norms = self.input_col.list.sum()
        norms = pl.when(norms == 0).then(1).otherwise(norms)
        return self.input_col.list.eval(pl.element() / norms).alias(self.output_col)

    @overload
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def __call__(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def __call__(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        return data.with_columns(self.polars_expr)

    @calidate_call
    def fixed_point_quantization(self, quantizer: FixedPointQuantizer) -> 'FixedPointQuantizedAlternativeNorm1':
        return FixedPointQuantizedAlternativeNorm1(
            input_col=self.input_col,
            output_col=self.output_col,
            quantizer=quantizer
        )


class FixedPointQuantizedAlternativeNorm1(AlternativeNorm1):

    quantizer: FixedPointQuantizer = Field(
        description="Fixed-point quantizer configuration."
    )

    @cached_property
    def polars_expr(self) -> pl.Expr:
        norm1_expr = super().polars_expr
        quantized_norm1_expr = norm1_expr.list.eval(pl.element().pipe(self.quantizer.quantize_polars_expr))
        return quantized_norm1_expr

    @overload
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def __call__(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def __call__(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        return data.with_columns(self.polars_expr)

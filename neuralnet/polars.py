import polars as pl
import numpy as np
from typing import Annotated, overload
from pydantic import Field, AfterValidator, BaseModel, ConfigDict
from functools import cached_property
from .quantization.quantizers import FixedPointQuantizer
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

    input_col: Annotated[
        list[str],
        Field(
            description="Name of the input column containing the rings data.",
            min_length=1,
        ),
    ]
    output_col_base: Annotated[
        str,
        Field(
            description="Base name for the output column containing the normalized rings data."
        ),
    ]

    @cached_property
    def output_cols(self) -> list[str]:
        if len(self.input_col) == 1:
            return [self.output_col_base]
        else:
            return [
                self.get_expanded_column_name(i) for i in range(len(self.input_col))
            ]

    def get_expanded_column_name(self, i: int) -> str:
        return f"{self.output_col_base}.{i}"

    def get_list_polars_expr(self) -> list[pl.Expr]:
        col = pl.col(self.input_col[0])
        norms = col.list.sum().abs()
        norms = pl.when(norms == 0).then(1).otherwise(norms)
        return [(col / norms).alias(self.output_col_base)]

    def get_arr_polars_expr(self) -> list[pl.Expr]:
        col = pl.col(self.input_col[0])
        norms = col.arr.sum().abs()
        norms = pl.when(norms == 0).then(1).otherwise(norms)
        return [(col / norms).alias(self.output_col_base)]

    def get_float_polars_expr(self) -> list[pl.Expr]:
        norms = pl.sum_horizontal(*self.input_col).abs()
        norms = pl.when(norms == 0).then(1).otherwise(norms)
        return [
            pl.col(col_name).truediv(norms).alias(self.get_expanded_column_name(i))
            for i, col_name in enumerate(self.input_col)
        ]

    def get_expr(self, df: pl.DataFrame | pl.LazyFrame) -> list[pl.Expr]:

        if isinstance(df, pl.DataFrame):
            schema = df.schema
        elif isinstance(df, pl.LazyFrame):
            schema = df.collect_schema()
        else:
            raise TypeError(f"Expected pl.DataFrame or pl.LazyFrame, got {type(df)}")

        if len(self.input_col) == 1:
            col_name = self.input_col[0]
            if col_name not in schema:
                raise ValueError(
                    f"Input column '{col_name}' not found in DataFrame schema."
                )
            col_dtype = schema[col_name]
            if (
                not isinstance(col_dtype, (pl.List, pl.Array))
                or not col_dtype.inner.is_float()
            ):
                raise TypeError(
                    f"Expected list or array column floats for '{col_name}', got {col_dtype}"
                )
            elif isinstance(col_dtype, pl.List):
                return self.get_list_polars_expr()
            elif isinstance(col_dtype, pl.Array):
                return self.get_arr_polars_expr()
            else:
                raise TypeError(
                    f"Expected list or array column for '{col_name}', got {col_dtype}"
                )

        for col_name in self.input_col:
            if col_name not in schema:
                raise ValueError(
                    f"Input column '{col_name}' not found in DataFrame schema."
                )
            col_dtype = schema[col_name]
            if not col_dtype.is_float():
                raise TypeError(
                    f"Expected floating type for '{col_name}', got {col_dtype}"
                )

        return self.get_float_polars_expr()

    @overload
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def __call__(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def __call__(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        expr = self.get_expr(data)
        return data.with_columns(*expr)

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

    def get_list_polars_expr(self) -> list[pl.Expr]:
        orig_expr = super().get_list_polars_expr()[0]
        new_expr = orig_expr.list.eval(
            pl.element().pipe(self.quantizer.quantize_polars_expr)
        )
        return [new_expr]

    def get_arr_polars_expr(self) -> list[pl.Expr]:
        orig_expr = super().get_arr_polars_expr()[0]
        new_expr = orig_expr.arr.eval(
            pl.element().pipe(self.quantizer.quantize_polars_expr)
        )
        return [new_expr]

    def get_float_polars_expr(self) -> list[pl.Expr]:
        orig_exprs = super().get_float_polars_expr()
        new_exprs = [
            expr.pipe(self.quantizer.quantize_polars_expr) for expr in orig_exprs
        ]
        return new_exprs

    @overload
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def __call__(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def __call__(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        expr = self.get_expr(data)
        return data.with_columns(expr)

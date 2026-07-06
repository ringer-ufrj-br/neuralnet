from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, overload
import polars as pl
from functools import cached_property
from ..quantization.quantizers import FixedPointQuantizer


class AlternativeNorm1(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_cols: Annotated[
        list[str],
        Field(
            description="Name of the input column containing the rings data.",
            min_length=1,
        ),
    ]
    suffix: Annotated[
        str,
        Field(
            description="Suffix for the output column containing normalized values."
        ),
    ] = "alternative_norm1"

    @cached_property
    def output_cols(self) -> list[str]:
        return [self.get_expanded_column_name(i) for i in range(len(self.input_cols))]

    def get_expanded_column_name(self, i: int) -> str:
        return f"{self.input_cols[i]}_{self.suffix}"

    def get_polars_expr(self) -> list[pl.Expr]:
        norms = pl.sum_horizontal(*self.input_cols, ignore_nulls=False).abs()
        norms = pl.when(norms == 0).then(1).otherwise(norms)
        return [
            pl.col(col_name).truediv(norms).alias(self.get_expanded_column_name(i))
            for i, col_name in enumerate(self.input_cols)
        ]

    def get_expr(self, df: pl.DataFrame | pl.LazyFrame) -> list[pl.Expr]:

        if isinstance(df, pl.DataFrame):
            schema = df.schema
        elif isinstance(df, pl.LazyFrame):
            schema = df.collect_schema()
        else:
            raise TypeError(f"Expected pl.DataFrame or pl.LazyFrame, got {type(df)}")

        for col_name in self.input_cols:
            if col_name not in schema:
                raise ValueError(
                    f"Input column '{col_name}' not found in DataFrame schema."
                )
            col_dtype = schema[col_name]
            if not col_dtype.is_float():
                raise TypeError(
                    f"Expected floating type for '{col_name}', got {col_dtype}"
                )

        return self.get_polars_expr()
    
    def fixed_point_quantization(
        self, quantizer: FixedPointQuantizer
    ) -> "FixedPointQuantizedAlternativeNorm1":
        return FixedPointQuantizedAlternativeNorm1(
            input_cols=self.input_cols, suffix=self.suffix, quantizer=quantizer
        )

    @overload
    def __call__(
        self, data: pl.DataFrame, passthrough: bool = False
    ) -> pl.DataFrame: ...

    @overload
    def __call__(
        self, data: pl.LazyFrame, passthrough: bool = False
    ) -> pl.LazyFrame: ...

    def __call__(
        self, data: pl.DataFrame | pl.LazyFrame, passthrough: bool = False
    ) -> pl.DataFrame | pl.LazyFrame:
        expr = self.get_expr(data)
        if passthrough:
            return data.with_columns(*expr)
        return data.select(*expr)

    def fixed_point_quantize(
        self, quantizer: FixedPointQuantizer
    ) -> "FixedPointQuantizedAlternativeNorm1":
        return FixedPointQuantizedAlternativeNorm1(
            input_cols=self.input_cols, output_col=self.output_col, quantizer=quantizer
        )


class FixedPointQuantizedAlternativeNorm1(AlternativeNorm1):
    quantizer: FixedPointQuantizer = Field(
        description="Fixed-point quantizer configuration."
    )

    def get_polars_expr(self) -> list[pl.Expr]:
        orig_exprs = super().get_polars_expr()
        new_exprs = [
            expr.pipe(self.quantizer.quantize_polars_expr) for expr in orig_exprs
        ]
        return new_exprs

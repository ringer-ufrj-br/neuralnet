import polars as pl
import numpy as np
from typing import Annotated
from pydantic import Field, AfterValidator

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

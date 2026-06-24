from functools import cached_property
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import polars as pl


class FixedPointQuantizer(BaseModel):
    integer_bits: Annotated[
        int,
        Field(
            ...,
            gt=0,
            description="Number of integer bits for fixed-point quantization."
        )
    ] = 1
    fractional_bits: Annotated[
        int,
        Field(
            ...,
            gt=0,
            description="Number of fractional bits for fixed-point quantization."
        )
    ]

    @cached_property
    def lower_bound(self) -> float:
        return -2 ** (self.integer_bits)

    @cached_property
    def upper_bound(self) -> float:
        return (2 ** self.integer_bits) - (2 ** -self.fractional_bits)

    @cached_property
    def floating_power(self) -> float:
        return 2 ** self.fractional_bits

    def quantize_polars_expr(self, expr: pl.Expr) -> pl.Expr:
        return (expr * self.floating_power).round().clip(self.lower_bound, self.upper_bound).truediv(self.floating_power)

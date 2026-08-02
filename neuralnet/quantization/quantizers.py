"""Quantizer models and helpers for quantization methods.

This module defines reusable quantizer configuration classes
and convenience helpers for applying fixed-point quantization.
"""

from functools import cached_property
from pydantic import BaseModel, Field
from typing import Annotated
import polars as pl


class FixedPointQuantizer(BaseModel):
    """Configuration for fixed-point quantization of numeric values.
    """

    integer_bits: Annotated[
        int,
        Field(
            ...,
            gt=0,
            description="Number of integer bits for fixed-point quantization.",
        ),
    ] = 1
    fractional_bits: Annotated[
        int,
        Field(
            ...,
            gt=0,
            description="Number of fractional bits for fixed-point quantization.",
        ),
    ]

    @cached_property
    def lower_bound(self) -> float:
        """Return the minimum representable fixed-point value.

        Returns
        -------
        float
            Lower clipping bound used before quantization.
        """
        return -(2 ** (self.integer_bits))

    @cached_property
    def upper_bound(self) -> float:
        """Return the maximum representable fixed-point value.

        Returns
        -------
        float
            Upper clipping bound used before quantization.
        """
        return (2**self.integer_bits) - (2**-self.fractional_bits)

    @cached_property
    def floating_power(self) -> float:
        """Return the power-of-two scaling factor used for quantization.

        Returns
        -------
        float
            Scaling factor equal to :math:`2^{\mathrm{fractional\_bits}}`.
        """
        return 2**self.fractional_bits

    def quantize_polars_expr(self, expr: pl.Expr) -> pl.Expr:
        """Quantize a Polars expression to the configured fixed-point grid.

        Parameters
        ----------
        expr : polars.Expr
            Expression whose values should be clipped and quantized.

        Returns
        -------
        polars.Expr
            Expression that evaluates to fixed-point quantized values.
        """
        return (
            expr
            .mul(self.floating_power)
            .round()
            .truediv(self.floating_power)
            .clip(self.lower_bound, self.upper_bound)
        )

from pydantic import BaseModel, Field
from typing import Annotated, Literal

from .polars import FixedPointQuantizedAlternativeNorm1


class FixedPointAlternativeNormL1(BaseModel):
    integer_bits: Annotated[
        int,
        Field(
            ...,
            gt=0,
            description="Number of integer bits for fixed-point quantization.",
        ),
    ]
    fractional_bits: Annotated[
        int,
        Field(
            ...,
            gt=0,
            description="Number of fractional bits for fixed-point quantization.",
        ),
    ]

    object_type: Annotated[
        Literal["fixed_point_alternative_norm1"],
        Field(description='Layer name. Must be "fixed_point_alternative_norm1"'),
    ] = "fixed_point_alternative_norm1"

    def as_polars_transform(self, input_cols: list[str]) -> FixedPointQuantizedAlternativeNorm1:
        """Create a FixedPointQuantizer instance from the configuration.

        Returns
        -------
        FixedPointQuantizedAlternativeNorm1
            An instance of FixedPointQuantizedAlternativeNorm1 with the specified integer and fractional bits.
        """
        return FixedPointQuantizedAlternativeNorm1(
            input_cols=input_cols,
            quantizer={
                "integer_bits": self.integer_bits,
                "fractional_bits": self.fractional_bits,
            },
        )

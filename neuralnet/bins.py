from functools import cached_property
from typing import Any, Literal, TypedDict, overload, Annotated
from pydantic import BaseModel, Field, BeforeValidator, PlainSerializer
import polars as pl
import numpy as np
from .numpy import Numpy1DFloatArray


def infinity_validator(value: Any) -> float:
    if value == "inf" or value == "Infinity":
        return float("inf")
    elif value == "-inf" or value == "-Infinity":
        return float("-inf")
    return float(value)


InfinityValidator = BeforeValidator(infinity_validator)


def infinity_serializer(value: float) -> str | float:
    if value == float("inf"):
        return "Infinity"
    elif value == float("-inf"):
        return "-Infinity"
    return value


InfinitySerializer = PlainSerializer(infinity_serializer, return_type=str | float)


type BinLowType = Annotated[
    float,
    Field(
        description="Lower bound of the bin. Can be a float or a string representing infinity ('inf', '-inf', 'Infinity', '-Infinity')."
    ),
    InfinityValidator,
    InfinitySerializer,
]

type BinHighType = Annotated[
    float,
    Field(
        description="Upper bound of the bin. Can be a float or a string representing infinity ('inf', '-inf', 'Infinity', '-Infinity')."
    ),
    InfinityValidator,
    InfinitySerializer,
]

type BinClosedType = Annotated[
    Literal["left", "right", "both", "none"],
    Field(
        description="Indicates whether the bin is closed on the left, right, both, or neither side."
    ),
]


class VariableBin(BaseModel):
    var_name: Annotated[
        str, Field(description="Name of the variable used in the binning.")
    ]
    low: BinLowType
    high: BinHighType
    closed: BinClosedType

    def as_polars_expr(self) -> "pl.Expr":
        return pl.col(self.var_name).is_between(self.low, self.high, closed=self.closed)

    @overload
    def apply_bin(self, df: "pl.DataFrame") -> "pl.DataFrame": ...

    @overload
    def apply_bin(self, df: "pl.LazyFrame") -> "pl.LazyFrame": ...

    def is_inside_polars(
        self, df: "pl.DataFrame | pl.LazyFrame"
    ) -> "pl.DataFrame | pl.LazyFrame":
        return df.filter(self.as_polars_expr())


class Bin(BaseModel):
    low: BinLowType
    high: BinHighType
    closed: BinClosedType

    def as_polars_expr(self, name: str) -> "pl.Expr":
        return pl.col(name).is_between(self.low, self.high, closed=self.closed)

    def sample(self, n_samples: int) -> Numpy1DFloatArray:
        """
        Sample values from the bin.

        Args:
            n_samples (int): Number of samples to draw.
        """
        if self.closed == "right" or self.closed == "none":
            left_limit = np.nextafter(self.low, np.inf)
        else:
            left_limit = self.low

        if self.closed == "left" or self.closed == "none":
            right_limit = np.nextafter(self.high, -np.inf)
        else:
            right_limit = self.high

        return np.random.uniform(left_limit, right_limit, size=n_samples)


class BinDict(TypedDict):
    low: BinLowType
    high: BinHighType
    closed: BinClosedType


type AbsoluteBinLowType = Annotated[
    float,
    Field(
        ge=0,
        description="Lower bound of the bin. Must be a non-negative float or a string representing infinity ('inf', 'Infinity').",
    ),
    InfinityValidator,
    InfinitySerializer,
]

type AbsoluteBinHighType = Annotated[
    float,
    Field(
        ge=0,
        description="Upper bound of the bin. Must be a non-negative float or a string representing infinity ('inf', 'Infinity').",
    ),
    InfinityValidator,
    InfinitySerializer,
]


class AbsoluteVariableBin(VariableBin):
    low: AbsoluteBinLowType
    high: AbsoluteBinHighType

    def as_polars_expr(self) -> pl.Expr:
        return (
            pl.col(self.var_name)
            .abs()
            .is_between(self.low, self.high, closed=self.closed)
        )


class AbsoluteBin(Bin):
    low: AbsoluteBinLowType
    high: AbsoluteBinHighType

    def as_polars_expr(self, name: str) -> "pl.Expr":
        return pl.col(name).abs().is_between(self.low, self.high, closed=self.closed)


class AbsoluteBinDict(TypedDict):
    low: AbsoluteBinLowType
    high: AbsoluteBinHighType
    closed: BinClosedType

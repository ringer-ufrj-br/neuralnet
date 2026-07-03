from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated, overload, Literal
from functools import cached_property
import polars as pl
from . import get_ring_slices_per_layer


class RingSlicesPerLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fraction: Annotated[
        int, Field(gt=0, description="Fraction of rings to select per layer.")
    ]
    output_format: Annotated[
        Literal["expanded_columns"],
        Field(
            description="Output format for the selected rings. Currently, only 'expanded_columns' is supported."
        ),
    ] = "expanded_columns"
    input_col: Annotated[
        str,
        Field(
            description="Name of the input column containing the list or array of rings."
        ),
    ]
    output_col_base: Annotated[
        str,
        Field(
            description="Name of the output column.",
        ),
    ]

    def get_expanded_column_name(self, i: int) -> str:
        return f"{self.output_col_base}.{i}"

    def get_list_polars_expr(self) -> pl.Expr:
        idxs = get_ring_slices_per_layer(self.fraction)
        if self.output_format == "expanded_columns":
            cols = [
                pl.col(self.input_col)
                .list.get(i)
                .alias(self.get_expanded_column_name(i))
                for i in idxs
            ]
            return cols
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format}")

    def get_array_polars_expr(self) -> pl.Expr:
        idxs = get_ring_slices_per_layer(self.fraction)
        if self.output_format == "expanded_columns":
            cols = [
                pl.col(self.input_col)
                .arr.get(i)
                .alias(self.get_expanded_column_name(i))
                for i in idxs
            ]
            return cols
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format}")

    @cached_property
    def output_cols(self) -> list[str]:
        idxs = get_ring_slices_per_layer(self.fraction)
        return [self.get_expanded_column_name(i) for i in idxs]

    @overload
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def __call__(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def __call__(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        if isinstance(data, pl.DataFrame):
            schema = data.schema
        elif isinstance(data, pl.LazyFrame):
            schema = data.collect_schema()
        else:
            raise TypeError(f"Expected pl.DataFrame or pl.LazyFrame, got {type(data)}")

        if self.input_col not in schema:
            raise ValueError(
                f"Input column '{self.input_col}' not found in the data schema."
            )

        input_dtype = schema[self.input_col]
        if isinstance(input_dtype, pl.List):
            polars_expr = self.get_list_polars_expr()
        elif isinstance(input_dtype, pl.Array):
            if not input_dtype.inner.is_float():
                raise TypeError(
                    f"Expected array of floats for column '{self.input_col}', got {input_dtype.inner}"
                )
            polars_expr = self.get_array_polars_expr()
        else:
            raise TypeError(f"Expected list or array column, got {input_dtype}")

        return data.with_columns(polars_expr)


class Open1DArray(BaseModel):
    input_col: Annotated[
        str,
        Field(description="Name of the input column containing the data array."),
    ]
    output_col: Annotated[
        str,
        Field(
            description="Name of the output column.",
        ),
    ]

    @overload
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def __call__(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def __call__(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        if isinstance(data, pl.DataFrame):
            schema = data.schema
        elif isinstance(data, pl.LazyFrame):
            schema = data.collect_schema()
        else:
            raise TypeError(f"Expected pl.DataFrame or pl.LazyFrame, got {type(data)}")

        if self.input_col not in schema:
            raise ValueError(
                f"Input column '{self.input_col}' not found in the data schema."
            )

        input_dtype = schema[self.input_col]
        if not isinstance(input_dtype, pl.Array):
            raise TypeError(
                f"Expected array for column '{self.input_col}', got {input_dtype}"
            )

        if not isinstance(input_dtype.shape, int):
            raise TypeError(
                f"Expected 1D array for column '{self.input_col}', got {input_dtype.shape}"
            )

        return data.with_columns(
            pl.col(self.input_col)
            .arr.to_struct(fields=lambda idx: f"{self.output_col}.{idx}")
            .alias(self.output_col)
        ).unnest(self.output_col)

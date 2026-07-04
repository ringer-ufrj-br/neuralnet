from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated, overload, Literal
from functools import cached_property
import polars as pl
from . import get_ring_slices_per_layer
from .. import get_logger


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
    rings_col: Annotated[
        str,
        Field(description="Name of the column containing the list or array of rings."),
    ]

    @cached_property
    def input_cols(self) -> list[str]:
        return [self.rings_col]

    @cached_property
    def output_cols(self) -> list[str]:
        idxs = get_ring_slices_per_layer(self.fraction)
        return [self.get_expanded_column_name(i) for i in idxs]

    def get_expanded_column_name(self, i: int) -> str:
        return f"{self.rings_col}.{i}"

    def get_list_polars_expr(self) -> pl.Expr:
        idxs = get_ring_slices_per_layer(self.fraction)
        if self.output_format == "expanded_columns":
            cols = [
                pl.col(self.rings_col)
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
                pl.col(self.rings_col)
                .arr.get(i)
                .alias(self.get_expanded_column_name(i))
                for i in idxs
            ]
            return cols
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format}")

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
        if isinstance(data, pl.DataFrame):
            schema = data.schema
        elif isinstance(data, pl.LazyFrame):
            schema = data.collect_schema()
        else:
            raise TypeError(f"Expected pl.DataFrame or pl.LazyFrame, got {type(data)}")

        if self.rings_col not in schema:
            raise ValueError(
                f"Input column '{self.rings_col}' not found in the data schema."
            )

        input_dtype = schema[self.rings_col]
        if isinstance(input_dtype, pl.List):
            polars_expr = self.get_list_polars_expr()
        elif isinstance(input_dtype, pl.Array):
            if not input_dtype.inner.is_float():
                raise TypeError(
                    f"Expected array of floats for column '{self.rings_col}', got {input_dtype.inner}"
                )
            polars_expr = self.get_array_polars_expr()
        else:
            raise TypeError(f"Expected list or array column, got {input_dtype}")

        if passthrough:
            return data.with_columns(polars_expr)
        else:
            return data.select(polars_expr)


def compute_class_weight(
    classes: list[int],
    df: pl.LazyFrame | pl.DataFrame,
    label_col: str = "label",
) -> dict[int, float]:
    """
    Computes class weights based on the distribution of classes in the dataset.

    Args:
        classes (list[int]): List of unique class labels.
        df (LazyFrame | DataFrame): The dataset containing the class labels.

    Returns:
        dict[int, float]: A dictionary mapping each class label to its corresponding weight.
    """
    class_counts_df = (
        df.select(label_col).group_by(label_col).len(name="count").collect()
    )
    class_counts = {
        int(row[label_col]): row["count"]
        for row in class_counts_df.iter_rows(named=True)
    }
    logger = get_logger()
    for class_ in classes:
        if class_ not in class_counts:
            logger.warning(f"Class {class_} not found.")
            class_counts[class_] = 0

    total_samples = sum(class_counts.values())
    n_classes = len(classes)
    class_weights = {
        class_: total_samples / (n_classes * count) if count > 0 else 1.0
        for class_, count in class_counts.items()
    }
    return class_weights

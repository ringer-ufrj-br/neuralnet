from typing import Annotated, Literal
from pydantic import Field, BaseModel
import polars as pl

from . import ParquetDataset


def get_ring_slices_per_layer(fraction: int) -> list[int]:
    # We select 1/fraction of rings in each layer
    # pre-sample - 8 rings
    # EM1 - 64 rings
    # EM2 - 8 rings
    # EM3 - 8 rings
    # Had1 - 4 rings
    # Had2 - 4 rings
    # Had3 - 4 rings
    rings_indexes = []
    # rings presmaple
    rings_indexes += list(range(8 // fraction))

    # EM1 list
    sum_rings = 8
    rings_indexes += list(range(sum_rings, sum_rings + (64 // fraction)))

    # EM2 list
    sum_rings = 8 + 64
    rings_indexes += list(range(sum_rings, sum_rings + (8 // fraction)))

    # EM3 list
    sum_rings = 8 + 64 + 8
    rings_indexes += list(range(sum_rings, sum_rings + (8 // fraction)))

    # HAD1 list
    sum_rings = 8 + 64 + 8 + 8
    rings_indexes += list(range(sum_rings, sum_rings + (4 // fraction)))

    # HAD2 list
    sum_rings = 8 + 64 + 8 + 8 + 4
    rings_indexes += list(range(sum_rings, sum_rings + (4 // fraction)))

    # HAD3 list
    sum_rings = 8 + 64 + 8 + 8 + 4 + 4
    rings_indexes += list(range(sum_rings, sum_rings + (4 // fraction)))

    return rings_indexes


class Bin(BaseModel):
    low: float = Field(..., description="Lower bound of the bin")
    high: float = Field(..., description="Upper bound of the bin")
    closed: str = Field(
        "left",
        description='Whether the bin is closed on the "left" or "right".',
    )


class RingerParquetDataset(ParquetDataset):
    data_table: Annotated[
        str, Field(description="Name of the data table in the parquet dataset")
    ]
    rings_col: Annotated[
        str, Field(description="Name of the rings column in the data table")
    ]
    kfold_table: Annotated[
        str, Field(description="Name of the kfold table in the parquet dataset")
    ]
    label_col: Annotated[
        str, Field(description="Name of the label column in the kfold table")
    ]
    fold_col: Annotated[
        str, Field(description="Name of the fold column in the kfold table")
    ]
    et_col: Annotated[str, Field(description="Name of the et column in the data table")]
    et_bin: Annotated[Bin | None, Field(None, description="Definition of the et bin")]
    eta_col: Annotated[
        str, Field(description="Name of the eta column in the data table")
    ]
    eta_bin: Annotated[Bin | None, Field(None, description="Definition of the eta bin")]
    ring_fraction: Annotated[
        int,
        Field(
            2,
            description="Fraction of the rings to be used for training. If 2, takes the first half of the rings for each layer. If 3, takes the first third of the rings, and so on.",
        ),
    ]
    kind: Literal['ringer_dataset'] = Field(
        'ringer_dataset',
        description="Kind of the dataset. Should be 'ringer_dataset' for this class.",
    )

    def get_n_folds(self) -> int:
        n_folds = (
            pl.scan_parquet(self.get_table_glob(self.kfold_table))
            .filter(pl.col(self.fold_col).is_not_null())
            .select(pl.col(self.fold_col).max().alias("max_fold"))
            .collect()
            .item()
        )
        return n_folds + 1  # Folds are 0-indexed

    def get_data_filter(self) -> pl.Expr | None:
        filters = []
        if self.et_bin is not None:
            et_col_expr = pl.col(self.et_col)
            filters.append(et_col_expr.is_between(
                pl.lit(self.et_bin.low, dtype=pl.dtype_of(et_col_expr)),
                pl.lit(self.et_bin.high, dtype=pl.dtype_of(et_col_expr)),
                closed=self.et_bin.closed
            ))
        if self.eta_bin is not None:
            eta_col_expr = pl.col(self.eta_col)
            filters.append(eta_col_expr.abs().is_between(
                pl.lit(self.eta_bin.low, dtype=pl.dtype_of(eta_col_expr)),
                pl.lit(self.eta_bin.high, dtype=pl.dtype_of(eta_col_expr)),
                closed=self.eta_bin.closed
            ))
        if filters:
            return pl.all_horizontal(*filters)
        return None
    
    def get_rings_expr(self):
        rings = []
        for ring_idx in get_ring_slices_per_layer(self.ring_fraction):
            rings.append(pl.col(self.rings_col).list.get(ring_idx).alias(f"ring_{ring_idx}"))
        return rings

    def get_fold_data(self, fold: int) -> tuple[pl.LazyFrame, pl.LazyFrame]:

        kfold_df = pl.scan_parquet(self.get_table_glob(self.kfold_table))

        data_df = pl.scan_parquet(self.get_table_glob(self.data_table))
        if data_filter := self.get_data_filter():
            data_df = data_df.filter(data_filter)
        data_df = data_df.select('id', *self.get_rings_expr())

        
        label = pl.col(self.label_col).alias('label')
        fold_col = pl.col(self.fold_col)
        fold = pl.lit(fold, dtype=pl.dtype_of(fold_col))
        val_fold_df = (
            kfold_df
            .filter((fold_col == fold) & label.is_not_null())
            .select("id", label.cast(pl.Int32))
        )
        train_fold_df = (
            kfold_df
            .filter((fold_col != fold) & label.is_not_null())
            .select("id", label.cast(pl.Int32))
        )

        train_df = data_df.join(train_fold_df, on="id", how="inner").drop("id")
        val_df = data_df.join(val_fold_df, on="id", how="inner").drop("id")

        return train_df, val_df

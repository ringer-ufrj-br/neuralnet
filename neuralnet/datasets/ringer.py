from functools import cached_property
from typing import Annotated, Literal, ClassVar
from pydantic import ConfigDict, Field, BaseModel, PrivateAttr
import polars as pl
from torch.utils.data import DataLoader
import numpy as np
import numpy.typing as npt

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
    model_config = ConfigDict(frozen=True)

    LABEL_COL: ClassVar[Literal["label"]] = "label"

    batch_size: int = Field(32, gt=0, description="Dataset batch size")
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
    object_type: Literal["ringer_parquet_dataset"] = Field(
        "ringer_parquet_dataset",
        description="Name of the dataset. Should be 'ringer_parquet_dataset' for this class.",
    )
    norm_strategy: Literal['l1'] | None = Field(
        None,
        description="Normalization strategy to apply to the rings. If None, no normalization is applied. If 'l1', each ring is divided by the sum of all rings for that sample.",
    )
    limit: int | None = Field(
        None,
        gt=0,
        description="Limit the number of samples to use from the dataset. Useful for debugging and testing.",
    )

    _fold: int = PrivateAttr(0)

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
            filters.append(
                et_col_expr.is_between(
                    pl.lit(self.et_bin.low, dtype=pl.dtype_of(et_col_expr)),
                    pl.lit(self.et_bin.high, dtype=pl.dtype_of(et_col_expr)),
                    closed=self.et_bin.closed,
                )
            )
        if self.eta_bin is not None:
            eta_col_expr = pl.col(self.eta_col)
            filters.append(
                eta_col_expr.abs().is_between(
                    pl.lit(self.eta_bin.low, dtype=pl.dtype_of(eta_col_expr)),
                    pl.lit(self.eta_bin.high, dtype=pl.dtype_of(eta_col_expr)),
                    closed=self.eta_bin.closed,
                )
            )
        if filters:
            return pl.all_horizontal(*filters)
        return None

    def get_rings_expr(self):
        rings = []
        names = []
        for ring_idx in get_ring_slices_per_layer(self.ring_fraction):
            name = f"ring_{ring_idx}"
            names.append(name)
            rings.append(
                pl.col(self.rings_col).list.get(ring_idx).alias(name)
            )
        return rings, names
    
    @cached_property
    def rings_aliases(self) -> list[str]:
        return [f"ring_{ring_idx}" for ring_idx in get_ring_slices_per_layer(self.ring_fraction)]
    
    @cached_property
    def rings_exprs(self) -> list[pl.Expr]:
        rings = []
        for ring_idx, name in zip(get_ring_slices_per_layer(self.ring_fraction), self.rings_aliases):
            rings.append(
                pl.col(self.rings_col).list.get(ring_idx).alias(name)
            )
        return rings
    
    @cached_property
    def l1_norm(self) -> pl.Expr:
        rings_sum = pl.col(self.rings_col).list.sum().abs()
        l1_norm = pl.when(
            rings_sum != 0
        ).then(
            rings_sum
        ).otherwise(
            pl.lit(1.0, dtype=pl.dtype_of(rings_sum))
        )
        return l1_norm

    def get_fold_data(self, group: Literal['train', 'val', 'test', 'predict']) -> pl.LazyFrame:

        data_df = pl.scan_parquet(self.get_table_glob(self.data_table))
        data_filter = self.get_data_filter()
        if data_filter is not None:
            data_df = data_df.filter(data_filter)
        if self.limit is not None:
            data_df = data_df.limit(self.limit)
        match self.norm_strategy:
            case 'l1':
                l1_alias = self.l1_norm.alias("l1_norm")
                rings_exprs = [
                    expr.truediv(l1_alias) for expr in self.rings_exprs
                ]
            case None:
                rings_exprs = self.rings_exprs
            case _:
                raise ValueError(f"Invalid normalization strategy: {self.norm_strategy}. Must be one of None or 'l1'.")
        data_df = data_df.select("id", *rings_exprs)

        label = pl.col(self.label_col).alias(self.LABEL_COL)
        fold_col = pl.col(self.fold_col)
        fold_df = pl.scan_parquet(self.get_table_glob(self.kfold_table))
        fold = pl.lit(self._fold, dtype=pl.dtype_of(fold_col))
        match group:
            case 'train':
                fold_df = fold_df.filter((label.is_not_null()) & (fold_col != fold)).select("id", label)
            case 'val':
                fold_df = fold_df.filter((label.is_not_null()) & (fold_col == fold)).select("id", label)
            case 'test':
                fold_df = fold_df.filter(label.is_not_null()).select("id", label)
            case 'predict':
                fold_df = fold_df.select("id", label)
            case _:
                raise ValueError(f"Invalid group: {group}. Must be one of 'train', 'val', 'test', or 'predict'.")

        return_df = data_df.join(fold_df, on="id", how="inner").drop("id")

        return return_df


    def train_df(self) -> pl.LazyFrame:
        return self.get_fold_data('train')
    
    def train_numpy(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        train_df = self.train_df().collect()
        X = train_df.drop(self.LABEL_COL).to_numpy()
        y = train_df.select(self.LABEL_COL).to_numpy().flatten()
        return X, y

    def train_dataloader(self) -> DataLoader:
        train_df = (
            self.train_df()
            .collect()
            .to_torch(
                "dataset",
                label=self.LABEL_COL,
            )
        )
        return DataLoader(train_df, batch_size=self.batch_size, shuffle=True)

    def val_df(self) -> pl.LazyFrame:
        return self.get_fold_data('val')
    
    def val_numpy(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        val_df = self.val_df().collect()
        X = val_df.drop(self.LABEL_COL).to_numpy()
        y = val_df.select(self.LABEL_COL).to_numpy().flatten()
        return X, y

    def val_dataloader(self) -> DataLoader:
        val_df = (
            self.val_df()
            .collect()
            .to_torch(
                "dataset",
                label=self.LABEL_COL,
            )
        )
        return DataLoader(val_df, batch_size=self.batch_size, shuffle=False)
    
    def test_df(self) -> pl.LazyFrame:
        return self.get_fold_data('test')

    def test_numpy(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        test_df = self.test_df().collect()
        X = test_df.drop(self.LABEL_COL).to_numpy()
        y = test_df.select(self.LABEL_COL).to_numpy().flatten()
        return X, y
    
    def test_dataloader(self) -> DataLoader:
        test_df = (
            self.test_df()
            .collect()
            .to_torch(
                "dataset",
                label=self.LABEL_COL,
            )
        )
        return DataLoader(test_df, batch_size=self.batch_size, shuffle=False)

    def predict_numpy(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        predict_df = self.predict_df().collect()
        X = predict_df.drop(self.LABEL_COL).to_numpy()
        y = predict_df.select(self.LABEL_COL).to_numpy().flatten()
        return X, y

    def predict_df(self) -> pl.LazyFrame:
        return self.get_fold_data('predict')

    def predict_dataloader(self) -> DataLoader:
        predict_df = (
            self.predict_df()
            .collect()
            .to_torch(
                "dataset",
                label=self.LABEL_COL,
            )
        )
        return DataLoader(predict_df, batch_size=self.batch_size, shuffle=False)

    def set_fold(self, fold: int) -> None:
        n_folds = self.get_n_folds()
        if fold < 0 or fold >= n_folds:
            raise ValueError(
                f"Fold index out of range. Must be between 0 and {n_folds - 1}."
            )
        self._fold = fold

    @property
    def current_fold(self) -> int:
        return self._fold

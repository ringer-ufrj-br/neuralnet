from typing import Annotated, Literal, ClassVar
from pydantic import (
    ConfigDict,
    Field
)
import polars as pl
import numpy as np
import numpy.typing as npt
from ...datasets import ParquetDataset
from ...models.binned_committee import VariableBin, Infinityvalidator
from ... import get_logger


class AbsoluteBin(VariableBin):
    low: Annotated[
        float, Field(..., ge=0, description="Lower bound of the bin"), Infinityvalidator
    ]
    high: Annotated[
        float, Field(..., ge=0, description="Upper bound of the bin"), Infinityvalidator
    ]

    def as_polars_expr(self) -> pl.Expr:
        return pl.col(self.var_name).abs().is_between(
            self.lower, self.upper, closed=self.closed
        )


type BatchSizeType = Annotated[
    int,
    Field(
        gt=0,
        description="Batch size for the dataset. Must be a positive integer.",
    ),
]

type DataTableType = Annotated[
    str, Field(description="Name of the data table in the parquet dataset")
]

type RingsColType = Annotated[
    str, Field(description="Name of the rings column in the data table")
]

type KFoldTableType = Annotated[
    str, Field(description="Name of the kfold table in the parquet dataset")
]

type LabelColType = Annotated[
    str, Field(description="Name of the label column in the kfold table")
]

type FoldColType = Annotated[
    str, Field(description="Name of the fold column in the kfold table")
]

type EtColType = Annotated[
    str, Field(description="Name of the et column in the data table")
]

type EtBinType = Annotated[
    VariableBin | None,
    Field(description="Definition of the et bin")
]

type EtaColType = Annotated[
    str, Field(description="Name of the eta column in the data table")
]

type EtaBinType = Annotated[
    AbsoluteBin | None, Field(description="Definition of the eta bin")
]

type RingFractionType = Annotated[
    int,
    Field(
        description="Fraction of the rings to be used for training. If 2, takes the first half of the rings for each layer. If 3, takes the first third of the rings, and so on.",
    ),
]

type ObjectTypeType = Annotated[
    Literal["ringer_parquet_dataset"],
    Field(
        description="Name of the dataset. Should be 'ringer_parquet_dataset' for this class.",
    ),
]

type NormStrategyType = Annotated[
    Literal["l1"] | None,
    Field(
        description="Normalization strategy to apply to the rings. If None, no normalization is applied. If 'l1', each ring is divided by the sum of all rings for that sample.",
    ),
]

type LimitType = Annotated[
    int | None,
    Field(
        gt=0,
        description="Limit the number of samples to use from the dataset. Useful for debugging and testing.",
    ),
]

type DataGroupType = Literal["train", "val", "test", "predict"]


class RingerParquetDataset(ParquetDataset):
    model_config = ConfigDict(frozen=True)

    N_RINGS: ClassVar[int] = 100
    CLASSES: ClassVar[list[int]] = [0, 1]

    data_table: DataTableType
    rings_col: RingsColType
    kfold_table: KFoldTableType
    label_col: LabelColType = "label"
    fold_col: FoldColType = "kfold"
    fold: Annotated[int, Field(description="Fold number to use for training.", ge=0)] = 0

    et_bin: EtBinType = None
    eta_bin: EtaBinType = None

    def get_n_folds(self) -> int:
        n_folds = (
            self.get_dataframe(self.kfold_table)
            .filter(pl.col(self.fold_col).is_not_null())
            .select(pl.col(self.fold_col).max().alias("max_fold"))
            .collect()
            .item()
        )
        return n_folds + 1  # Folds are 0-indexed


    def open_rings_expr(self) -> tuple[list[pl.Expr], list[str]]:
        rings = []
        names = []
        for ring_idx in range(self.N_RINGS):
            name = f"{self.rings_col}.{ring_idx}"
            names.append(name)
            rings.append(pl.col(self.rings_col).list.get(ring_idx).alias(name))
        return rings, names


    def get_fold_data(self, group: DataGroupType) -> pl.LazyFrame:

        label = pl.col(self.label_col)
        fold_col = pl.col(self.fold_col)
        fold_df = self.get_dataframe(self.kfold_table)
        fold = pl.lit(self.fold, dtype=pl.dtype_of(fold_col))
        is_test_col = label.is_not_null() & fold_col.is_not_null()
        is_val_col = label.is_not_null() & fold_col == fold
        is_train_col = label.is_not_null() & fold_col != fold
        fold_df = fold_df.with_columns(
            is_test_col.alias("is_test"),
            is_val_col.alias("is_val"),
            is_train_col.alias("is_train"),
        )
        match group:
            case "train":
                fold_df = fold_df.filter(pl.col("is_train"))
            case "val":
                fold_df = fold_df.filter(pl.col("is_val"))
            case "test":
                fold_df = fold_df.filter(pl.col("is_test"))
            case "predict":
                pass
            case _:
                raise ValueError(
                    f"Invalid group: {group}. Must be one of 'train', 'val', 'test', or 'predict'."
                )

        data_df = self.get_dataframe(self.data_table)
        if self.et_bin is not None:
            data_df = self.et_bin.is_inside_polars(data_df)
        if self.eta_bin is not None:
            data_df = self.eta_bin.is_inside_polars(data_df)
        return_df = data_df.join(fold_df, on="id", how="inner")

        return return_df

    def get_class_weights(self, group: DataGroupType) -> dict[int, float]:
        match group:
            case "train":
                df = self.train_df()
            case "val":
                df = self.val_df()
            case "test":
                df = self.test_df()
            case "predict":
                df = self.predict_df()
            case _:
                raise ValueError(
                    f"Invalid group: {group}. Must be one of 'train', 'val', 'test', or 'predict'."
                )
        class_counts_df = (
            df.select(self.label_col)
            .group_by(self.label_col)
            .len(name="count")
            .collect()
        )
        class_counts = {
            int(row[self.label_col]): row["count"]
            for row in class_counts_df.iter_rows(named=True)
        }
        logger = get_logger()
        for class_ in self.CLASSES:
            if class_ not in class_counts:
                logger.warning(f"Class {class_} not found in {group} set")
                class_counts[class_] = 0
        total_samples = sum(class_counts.values())
        n_classes = len(self.CLASSES)
        class_weights = {
            class_: total_samples / (n_classes * count) if count > 0 else 1.0
            for class_, count in class_counts.items()
        }
        return class_weights

    def get_sample_weights_expr(self, group: DataGroupType) -> pl.Expr:
        class_weights = self.get_class_weights(group)
        weights_expr = (
            pl.col(self.label_col).replace(class_weights).alias("sample_weight")
        )
        return weights_expr

    def get_sample_weights(self, group: DataGroupType) -> pl.LazyFrame:
        expr = self.get_sample_weights_expr(group)
        return self.get_fold_data(group).select(expr)

    def train_df(self) -> pl.LazyFrame:
        return self.get_fold_data("train")

    def train_class_weights(self) -> dict[int, float]:
        return self.get_class_weights("train")

    def train_sample_weights(self) -> np.ndarray[tuple[int], np.floating]:
        return self.get_sample_weights("train").collect().to_numpy().flatten()

    def train_numpy(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        df = self.train_df()
        open_rings_expr, _ = self.open_rings_expr()
        X, y = pl.collect_all([df.select(open_rings_expr), df.select(self.label_col)])
        X = X.to_numpy()
        y = y.to_numpy().flatten()
        return X, y

    def train_dataloader(self, batch_size: int):
        from torch.utils.data import DataLoader

        open_rings_expr, features = self.open_rings_expr()
        train_df = (
            self.train_df()
            .select(*open_rings_expr, self.label_col)
            .collect()
            .to_torch("dataset", label=self.label_col, features=features)
        )
        return DataLoader(train_df, batch_size=batch_size, shuffle=True)

    def val_df(self) -> pl.LazyFrame:
        return self.get_fold_data("val")

    def val_class_weights(self) -> dict[int, float]:
        return self.get_class_weights("val")

    def val_sample_weights(self) -> np.ndarray[tuple[int], np.floating]:
        return self.get_sample_weights("val").collect().to_numpy().flatten()

    def val_numpy(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        df = self.val_df()
        open_rings_expr, _ = self.open_rings_expr()
        X, y = pl.collect_all([df.select(open_rings_expr), df.select(self.label_col)])
        X = X.to_numpy()
        y = y.to_numpy().flatten()
        return X, y

    def val_dataloader(self, batch_size: int):
        from torch.utils.data import DataLoader
        open_rings_expr, features = self.open_rings_expr()
        val_df = (
            self.val_df()
            .select(*open_rings_expr, self.label_col)
            .collect()
            .to_torch("dataset", label=self.label_col, features=features)
        )
        return DataLoader(val_df, batch_size=batch_size, shuffle=False)

    def test_df(self) -> pl.LazyFrame:
        return self.get_fold_data("test")

    def test_class_weights(self) -> dict[int, float]:
        return self.get_class_weights("test")

    def test_sample_weights(self) -> np.ndarray[tuple[int], np.floating]:
        return self.get_sample_weights("test").collect().to_numpy().flatten()

    def test_numpy(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        df = self.test_df()
        open_rings_expr, _ = self.open_rings_expr()
        X, y = pl.collect_all([df.select(open_rings_expr), df.select(self.label_col)])
        X = X.to_numpy()
        y = y.to_numpy().flatten()
        return X, y

    def test_dataloader(self, batch_size: int):
        from torch.utils.data import DataLoader
        open_rings_expr, features = self.open_rings_expr()
        test_df = (
            self.test_df()
            .select(*open_rings_expr, self.label_col)
            .collect()
            .to_torch("dataset", label=self.label_col, features=features)
        )
        return DataLoader(test_df, batch_size=batch_size, shuffle=False)

    def predict_class_weights(self) -> dict[int, float]:
        return self.get_class_weights("predict")

    def predict_sample_weights(self) -> np.ndarray[tuple[int], np.floating]:
        return self.get_sample_weights("predict").collect().to_numpy().flatten()

    def predict_numpy(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        df = self.predict_df()
        open_rings_expr, _ = self.open_rings_expr()
        X, y = pl.collect_all([df.select(open_rings_expr), df.select(self.label_col)])
        X = X.to_numpy()
        y = y.to_numpy().flatten()
        return X, y

    def predict_df(self) -> pl.LazyFrame:
        return self.get_fold_data("predict")

    def predict_dataloader(self, batch_size: int):
        from torch.utils.data import DataLoader
        open_rings_expr, features = self.open_rings_expr()
        predict_df = (
            self.predict_df()
            .select(*open_rings_expr, self.label_col)
            .collect()
            .to_torch("dataset", label=self.label_col, features=features)
        )
        return DataLoader(predict_df, batch_size=batch_size, shuffle=False)

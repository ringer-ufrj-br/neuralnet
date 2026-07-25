"""Utilities for working with ringer parquet datasets.

This module provides a Pydantic-backed dataset wrapper plus helpers for
building fold splits, sample weights, and synthetic ringer-like data.
"""

from typing import Annotated, Literal, ClassVar
from pydantic import ConfigDict, Field
import polars as pl
import numpy as np
from itertools import product
from . import ParquetDataset
from .. import get_logger
from ..bins import VariableBin, AbsoluteVariableBin, BinDict, Bin


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

type EtBinType = Annotated[
    VariableBin | None, Field(description="Definition of the et bin")
]

type EtaBinType = Annotated[
    AbsoluteVariableBin | None, Field(description="Definition of the eta bin")
]

type DataGroupType = Literal["train", "val", "test", "predict"]


class RingerParquetDataset(ParquetDataset):
    """Dataset wrapper for ringer parquet data and fold-based evaluation.

    The dataset stores the names of the parquet tables and columns required to
    access the rings features and the k-fold labels. Individual field
    descriptions are declared on the Pydantic annotations.

    The data is organized around a labeled k-fold table joined to the main data
    table by ``id``. The ``train`` and ``val`` splits are selected from the
    labeled rows using the configured fold index, while ``test`` is the union of
    those labeled rows. The ``predict`` split extends ``test`` with the
    unlabeled rows present in the original tables, allowing inference on both
    held-out labeled samples and unlabeled data.
    """

    model_config = ConfigDict(frozen=True)

    N_RINGS: ClassVar[int] = 100
    CLASSES: ClassVar[list[int]] = [0, 1]

    data_table: DataTableType
    rings_col: RingsColType
    kfold_table: KFoldTableType
    label_col: LabelColType = "label"
    fold_col: FoldColType = "kfold"
    fold: Annotated[
        int, Field(description="Fold number to use for training.", ge=0)
    ] = 0

    et_bin: EtBinType = None
    eta_bin: EtaBinType = None

    def get_n_folds(self) -> int:
        """Return the number of cross-validation folds in the dataset.

        Returns
        -------
        int
            Total number of folds, inferred from the maximum fold index in the
            k-fold table.
        """
        n_folds = (
            self.get_dataframe(self.kfold_table)
            .filter(pl.col(self.fold_col).is_not_null())
            .select(pl.col(self.fold_col).max().alias("max_fold"))
            .collect()
            .item()
        )
        return n_folds + 1  # Folds are 0-indexed

    def get_fold_data(self, group: DataGroupType) -> pl.LazyFrame:
        """Return the lazy frame for a specific dataset split.

        Parameters
        ----------
        group : {"train", "val", "test", "predict"}
            Dataset split to retrieve.

        Returns
        -------
        polars.LazyFrame
            Lazy frame containing the selected split joined with the data
            table.

        Raises
        ------
        ValueError
            If ``group`` is not one of the supported split names.
        """

        label = pl.col(self.label_col)
        fold_col = pl.col(self.fold_col)
        fold_df = self.get_dataframe(self.kfold_table)
        fold = pl.lit(self.fold, dtype=pl.dtype_of(fold_col))
        is_test_col = (label.is_not_null()) & (fold_col.is_not_null())
        is_val_col = (label.is_not_null()) & (fold_col == fold)
        is_train_col = (label.is_not_null()) & (fold_col != fold)
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
            data_df = data_df.pipe(self.et_bin.is_inside_polars)
        if self.eta_bin is not None:
            data_df = data_df.pipe(self.eta_bin.is_inside_polars)
        return_df = data_df.join(fold_df, on="id", how="inner")

        return return_df

    def get_class_weights(self, group: DataGroupType) -> dict[int, float]:
        """Compute inverse-frequency class weights for a split.

        Parameters
        ----------
        group : {"train", "val", "test", "predict"}
            Dataset split used to compute the class distribution.

        Returns
        -------
        dict[int, float]
            Mapping from class label to weight.

        Raises
        ------
        ValueError
            If ``group`` is not one of the supported split names.
        """
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
        """Return a Polars expression that maps labels to sample weights.

        Parameters
        ----------
        group : {"train", "val", "test", "predict"}
            Dataset split used to compute the class weights.

        Returns
        -------
        polars.Expr
            Expression that produces a ``sample_weight`` column.
        """
        class_weights = self.get_class_weights(group)
        weights_expr = pl.col(self.label_col).cast(pl.Float64).replace(class_weights).alias(
            "sample_weight"
        )
        return weights_expr

    def get_sample_weights(self, group: DataGroupType) -> pl.LazyFrame:
        """Return the sample-weight column for a dataset split.

        Parameters
        ----------
        group : {"train", "val", "test", "predict"}
            Dataset split used to compute the weights.

        Returns
        -------
        polars.LazyFrame
            Lazy frame containing the sample weights.
        """
        expr = self.get_sample_weights_expr(group)
        return self.get_fold_data(group).select(expr)

    def train_df(self) -> pl.LazyFrame:
        """Return the training split as a lazy frame."""
        return self.get_fold_data("train")

    def train_class_weights(self) -> dict[int, float]:
        """Return the training-split class weights."""
        return self.get_class_weights("train")

    def train_sample_weights(self) -> np.ndarray[tuple[int], np.floating]:
        """Return the training-split sample weights as a NumPy array."""
        return self.get_sample_weights("train").collect().to_numpy().flatten()

    def val_df(self) -> pl.LazyFrame:
        """Return the validation split as a lazy frame."""
        return self.get_fold_data("val")

    def val_class_weights(self) -> dict[int, float]:
        """Return the validation-split class weights."""
        return self.get_class_weights("val")

    def val_sample_weights(self) -> np.ndarray[tuple[int], np.floating]:
        """Return the validation-split sample weights as a NumPy array."""
        return self.get_sample_weights("val").collect().to_numpy().flatten()

    def test_df(self) -> pl.LazyFrame:
        """Return the test split as a lazy frame."""
        return self.get_fold_data("test")

    def test_class_weights(self) -> dict[int, float]:
        """Return the test-split class weights."""
        return self.get_class_weights("test")

    def test_sample_weights(self) -> np.ndarray[tuple[int], np.floating]:
        """Return the test-split sample weights as a NumPy array."""
        return self.get_sample_weights("test").collect().to_numpy().flatten()

    def predict_class_weights(self) -> dict[int, float]:
        """Return the prediction-split class weights."""
        return self.get_class_weights("predict")

    def predict_sample_weights(self) -> np.ndarray[tuple[int], np.floating]:
        """Return the prediction-split sample weights as a NumPy array."""
        return self.get_sample_weights("predict").collect().to_numpy().flatten()

    def predict_df(self) -> pl.LazyFrame:
        """Return the prediction split as a lazy frame."""
        return self.get_fold_data("predict")


def generate_ringer_dataset_dfs(
    et_bins: list[Bin | BinDict],
    eta_bins: list[Bin | BinDict],
    samples_per_bin: int = 1000,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate synthetic ringer-like data and fold assignment tables.

    Parameters
    ----------
    et_bins : list[Bin or BinDict]
        Bins used to generate the ``et`` feature values.
    eta_bins : list[Bin or BinDict]
        Bins used to generate the ``eta`` feature values.
    samples_per_bin : int, default=1000
        Number of synthetic samples to generate for each bin combination.
    n_folds : int, default=5
        Number of folds to draw when assigning fold labels.
    random_state : int, default=42
        Seed forwarded to the blob generator.

    Returns
    -------
    tuple[polars.DataFrame, polars.DataFrame]
        A pair containing the data table and the k-fold table, respectively.
    """
    from sklearn.datasets import make_blobs
    for i in range(len(et_bins)):
        et_bins[i] = et_bins[i] if isinstance(et_bins[i], Bin) else Bin(**et_bins[i])
    for i in range(len(eta_bins)):
        eta_bins[i] = eta_bins[i] if isinstance(eta_bins[i], Bin) else Bin(**eta_bins[i])
    
    final_data_df = []
    final_kfold_df = []
    id_starts = range(0, samples_per_bin * len(et_bins) * len(eta_bins), samples_per_bin)
    iterator = zip(
        id_starts,
        product(et_bins, eta_bins),
    )
    for id_start, (et_bin, eta_bin) in iterator:

        rings, labels = make_blobs(
            n_samples=samples_per_bin,
            n_features=RingerParquetDataset.N_RINGS,
            centers=2,
            cluster_std=0.1,
            random_state=random_state,
        )
        rings = rings.astype(np.float32)
        labels = labels.astype(np.bool_)
        data_df = pl.DataFrame(
            {
                "id": np.arange(id_start, id_start + samples_per_bin, dtype=np.uint64),
                "rings": [r for r in rings],
                "et": et_bin.sample(samples_per_bin).astype(np.float32),
                "eta": eta_bin.sample(samples_per_bin).astype(np.float32),
            }
        ).with_columns(pl.col("rings").cast(pl.List(pl.Float32)))
        final_data_df.append(data_df)

        kfold_df = pl.DataFrame(
            {
                "id": np.arange(id_start, id_start + samples_per_bin),
                "label": labels.tolist(),
                "fold": np.random.randint(0, n_folds, samples_per_bin).tolist(),
            }
        )
        final_kfold_df.append(kfold_df)
    
    final_kfold_df = pl.concat(final_kfold_df)
    final_data_df = pl.concat(final_data_df)

    return final_data_df, final_kfold_df
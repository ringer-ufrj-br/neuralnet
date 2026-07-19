"""Base dataset helpers for reading parquet-backed datasets.

This module provides the shared dataset abstraction used by the ringer
datasets and other parquet-based data sources in the project.
"""

from pathlib import Path
from typing import Annotated
from functools import cached_property

from pydantic import BaseModel, Field
import polars as pl


type RefType = dict[str, dict[str, dict[str, float]]]

type DirectoryType = Annotated[
    Path, Field(description="Path to the directory containing the dataset files.")
]


class ParquetDataset(BaseModel):
    """Base model for parquet-backed datasets.
    """

    dataset_dir: DirectoryType

    def get_table_glob(self, table_name: str) -> Path:
        """Return a glob pattern or file path for a dataset table.

        Parameters
        ----------
        table_name : str
            Table name with or without the ``.parquet`` suffix.

        Returns
        -------
        pathlib.Path
            Path to the parquet file or glob pattern matching parquet files in
            the table directory.
        """
        if table_name.endswith(".parquet"):
            base_path = Path(self.dataset_dir) / table_name
        else:
            base_path = Path(self.dataset_dir) / f"{table_name}.parquet"
        if base_path.is_file():
            return base_path
        return base_path / "*.parquet"

    def get_table_path(self, table_name: str) -> Path:
        """Return the canonical parquet path for a dataset table.

        Parameters
        ----------
        table_name : str
            Table name with or without the ``.parquet`` suffix.

        Returns
        -------
        pathlib.Path
            Path to the parquet file for the requested table.
        """
        if table_name.endswith(".parquet"):
            return Path(self.dataset_dir) / table_name
        return Path(self.dataset_dir) / f"{table_name}.parquet"

    def get_dataframe(self, table_name: str) -> pl.LazyFrame:
        """Open a parquet table as a Polars lazy frame.

        Parameters
        ----------
        table_name : str
            Table name with or without the ``.parquet`` suffix.

        Returns
        -------
        polars.LazyFrame
            Lazy frame scanning the requested parquet table.
        """
        return pl.scan_parquet(self.get_table_path(table_name))

    @cached_property
    def sql_context(self) -> pl.SQLContext:
        """Build a cached SQL context from every parquet file in the dataset.

        Returns
        -------
        polars.SQLContext
            SQL context with each top-level parquet file registered under its
            stem name.
        """
        dataframes = {
            table.stem: self.get_dataframe(table.stem)
            for table in self.dataset_dir.glob("*.parquet")
        }
        return pl.SQLContext(**dataframes)

from pathlib import Path
from typing import Annotated
import duckdb

from pydantic import BaseModel, Field
import typer


type RefType = dict[str, dict[str, dict[str, float]]]

type DirectoryType = Annotated[
    Path, Field(description="Path to the directory containing the dataset files.")
]


class ParquetDataset(BaseModel):
    dataset_dir: DirectoryType

    def get_table_glob(self, table_name: str) -> Path:
        if table_name.endswith(".parquet"):
            base_path = Path(self.dataset_dir) / table_name
        else:
            base_path = Path(self.dataset_dir) / f"{table_name}.parquet"
        if base_path.is_file():
            return base_path
        return base_path / "*.parquet"

    def get_table_path(self, table_name: str) -> Path:
        if table_name.endswith(".parquet"):
            return Path(self.dataset_dir) / table_name
        return Path(self.dataset_dir) / f"{table_name}.parquet"


app = typer.Typer()


@app.command()
def print_schema(
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset-dir", help="Directory containing the dataset files"),
    ],
):
    dataset = ParquetDataset(dataset_dir=dataset_dir)
    for table in dataset_dir.glob("*.parquet"):
        with duckdb.connect(":memory:") as conn:
            res = conn.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{str(dataset.get_table_glob(table.name))}')"
            ).fetch_df()
        print(20 * "-")
        print(f"Schema for {table.name}:")
        print(res.to_string())
        print(20 * "-")

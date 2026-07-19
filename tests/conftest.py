from pathlib import Path
from typing import Generator
import pytest
from concurrent.futures import ProcessPoolExecutor
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neuralnet.datasets.ringer import RingerParquetDataset


@pytest.fixture
def root_dir() -> Path:
    return Path(__file__).parent.parent.absolute()


@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parent.absolute() / "data"


@pytest.fixture(scope="session")
def create_logs_dir() -> Path:
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True, parents=True)
    return logs_dir


@pytest.fixture()
def isolated_executor() -> Generator[ProcessPoolExecutor, None, None]:
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as executor:
        yield executor


@pytest.fixture(scope="session")
def session_tmp_path():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="pytest_session_tmp_path_") as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope='session')
def ringer_dataset_dir(session_tmp_path: Path) -> Path:
    from neuralnet.datasets.ringer import generate_ringer_dataset_dfs

    logging.info(f"Creating Ringer dataset in {session_tmp_path}")

    dataset_dir = session_tmp_path / "ringer_dataset_dir"
    dataset_dir.mkdir()

    data_df, kfold_df = generate_ringer_dataset_dfs(
        et_bins=[
            {"low": 0.0, "high": 50000.0, "closed": "left"},
            {"low": 50000.0, "high": 100000.0, "closed": "left"}
        ],
        eta_bins=[
            {"low": 0, "high": 0.8, "closed": "left"},
            {"low": 0.8, "high": 1.37, "closed": "left"},
        ],
        samples_per_bin=1000,
        n_folds=2,
        random_state=42,
    )
    data_df.write_parquet(dataset_dir / "data.parquet")
    kfold_df.write_parquet(dataset_dir / "kfold.parquet")

    return dataset_dir

@pytest.fixture
def ringer_parquet_dataset(ringer_dataset_dir: Path) -> 'RingerParquetDataset':
    from neuralnet.datasets.ringer import RingerParquetDataset
    return RingerParquetDataset(
        dataset_dir=ringer_dataset_dir,
        data_table="data",
        rings_col="rings",
        kfold_table="kfold",
        label_col="label",
        fold_col="fold",
        fold=0,
    )

@pytest.fixture(scope="session")
def small_ringer_dataset_dir(session_tmp_path: Path) -> Path:
    import polars as pl
    import numpy as np
    tmp_dir = Path(session_tmp_path)

    logging.info(f"Creating small ringer dataset in {session_tmp_path}")

    dataset_dir = tmp_dir / "dataset"
    dataset_dir.mkdir()

    n_samples = 6
    n_rings = 100

    data_df = pl.DataFrame(
        {
            "id": np.arange(n_samples),
            "rings": [np.arange(n_rings, dtype=float).tolist() for _ in range(n_samples)],
            "et": np.linspace(10.0, 60.0, n_samples),
            "eta": np.linspace(0.1, 2.1, n_samples),
        }
    )
    data_df.write_parquet(dataset_dir / "data.parquet")

    kfold_df = pl.DataFrame(
        {
            "id": np.arange(n_samples),
            "label": [0, 1, 1, 0, 1, 0],
            "fold": [0, 0, 1, 1, 0, 1],
        }
    )
    kfold_df.write_parquet(dataset_dir / "kfold.parquet")

    return dataset_dir


@pytest.fixture
def small_ringer_parquet_dataset(small_ringer_dataset_dir: Path) -> 'RingerParquetDataset':
    from neuralnet.datasets.ringer import RingerParquetDataset
    return RingerParquetDataset(
        dataset_dir=small_ringer_dataset_dir,
        data_table="data",
        rings_col="rings",
        kfold_table="kfold",
        label_col="label",
        fold_col="fold",
        fold=0,
    )
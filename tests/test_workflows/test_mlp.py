import pytest
import polars as pl
import numpy as np
from pathlib import Path
import logging

from neuralnet.workflows.mlp.dataset import RingerParquetDataset


MLP_JOB_CONFIG = {
    "batch_size": 32,
    "data_table": "data",
    "et_col": "et",
    "et_bins": [{"low": 0.0, "high": 100.0, "closed": "left"}],
    "eta_col": "eta",
    "eta_bins": [{"low": 0.0, "high": 2.5, "closed": "left"}],
    "fold_col": "fold",
    "kfold_table": "kfold",
    "label_col": "target_label",
    "rings_col": "rings",
    "ring_fraction": 1,
    "model_factory": {
        "object_type": "mlp",
        "layers": [
            {"object_type": "dense", "units": 4, "activation": "relu"},
            {"object_type": "dense", "units": 1, "activation": "sigmoid"},
        ],
        "name": "mlp",
    },
    "loss": {"object_type": "binary_cross_entropy", "from_logits": False},
    "optimizer": {"learning_rate": 0.01, "object_type": "adam"},
    "from_logits": False,
    "epochs": 1,
    "logger_name": None,
    "output_path": None,  # To be set in the test
    "executor_config": None,  # To be set in the test
    "inits": 1,
}

TEST_DATA = {
    "mlp": MLP_JOB_CONFIG,
}


@pytest.fixture
def ringer_dataset_dir(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    n_samples = 6
    n_rings = RingerParquetDataset.N_RINGS

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
            "target_label": [0, 1, 1, 0, 1, 0],
            "fold": [0, 0, 1, 1, 0, 1],
        }
    )
    kfold_df.write_parquet(dataset_dir / "kfold.parquet")

    return dataset_dir


@pytest.fixture
def ringer_parquet_dataset(ringer_dataset_dir: Path) -> RingerParquetDataset:
    return RingerParquetDataset(
        dataset_dir=ringer_dataset_dir,
        data_table="data",
        rings_col="rings",
        kfold_table="kfold",
        label_col="target_label",
        fold_col="fold",
        fold=0,
    )


@pytest.mark.parametrize(
    ("job_config"),
    list(TEST_DATA.values()),
    ids=list(TEST_DATA.keys()),
)
def test_keras_ringer_committee_keras_training_job(
    tmp_path: Path,
    job_config: dict,
    ringer_dataset_dir: Path,
    ringer_parquet_dataset: RingerParquetDataset,
    isolated_executor,
):
    job_config = dict(job_config)
    future = isolated_executor.submit(
        keras_ringer_committee_keras_training_job,
        job_config=job_config,
        tmp_path=tmp_path,
        dataset_dir=ringer_dataset_dir,
        ringer_parquet_dataset=ringer_parquet_dataset
    )
    future.result()


def keras_ringer_committee_keras_training_job(
    tmp_path: Path, job_config: dict, dataset_dir: Path,ringer_parquet_dataset: RingerParquetDataset
):

    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    from neuralnet.workflows.mlp.jobs import (
        MLPKerasTrainingJob,
    )
    from neuralnet.workflows.mlp.dataset import RingerParquetDataset
    from neuralnet.submitit import ExecutorConfig
    job_config["dataset_dir"] = dataset_dir
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    executor_config = ExecutorConfig(
        cpus_per_task=1,
        executor_type="debug",
        logs_dir=logs_dir,
        name="test_job",
        slurm_partition="test",
    )
    job_config["executor_config"] = executor_config
    job_config["output_path"] = tmp_path / "output"

    job = MLPKerasTrainingJob(**job_config)
    job.submit()

    loaded_job = MLPKerasTrainingJob.load(job_config["output_path"])
    assert isinstance(loaded_job, MLPKerasTrainingJob), (
        "Loaded job is not of the expected type"
    )
    assert loaded_job == job, "Loaded job is not equal to the original job"
    MLPKerasTrainingJob.validate_saved_directory(job_config["output_path"])
    assert isinstance(loaded_job.all_model_results, pl.DataFrame), (
        "Loaded job's all_model_results is not a Polars DataFrame"
    )
    assert isinstance(loaded_job.selected_models, pl.DataFrame), (
        "Loaded job's selected_models is not a Polars DataFrame"
    )
    n_folds = ringer_parquet_dataset.get_n_folds()
    assert loaded_job.all_model_results.height == n_folds * job.inits * len(job.et_bins) * len(job.eta_bins)
    assert 1 <= loaded_job.selected_models.height <= loaded_job.all_model_results.height

    inference_pipeline = loaded_job.get_inference_pipeline()
    inference_dataset = RingerParquetDataset(
        dataset_dir=dataset_dir,
        data_table=job.data_table,
        rings_col=job.rings_col,
        kfold_table=job.kfold_table,
        label_col=job.label_col,
        fold_col=job.fold_col,
        fold=0,
    )
    inference_input = inference_dataset.predict_df()
    inference_results = inference_pipeline(inference_input).collect()

    assert inference_results.height == 1 * loaded_job.selected_models.height
    assert "id" in inference_results.columns
    assert "prediction" in inference_results.columns
    assert "output" in inference_results.columns

    assert inference_results.get_column("prediction").dtype == pl.Boolean
    assert inference_results.get_column("output").min() >= 0
    assert inference_results.get_column("output").max() <= 1

    logging.info("Finished")


def test_ringer_parquet_dataset_splits_and_weights(
    ringer_parquet_dataset: RingerParquetDataset,
):
    dataset = ringer_parquet_dataset

    assert dataset.get_n_folds() == 2

    rings_expr, ring_names = dataset.open_rings_expr()
    assert len(rings_expr) == dataset.N_RINGS
    assert len(ring_names) == dataset.N_RINGS
    assert ring_names[0] == "rings.0"
    assert ring_names[-1] == "rings.99"

    assert dataset.train_df().collect().get_column("id").to_list() == [2, 3, 5]
    assert dataset.val_df().collect().get_column("id").to_list() == [0, 1, 4]
    assert dataset.test_df().collect().get_column("id").to_list() == [0, 1, 2, 3, 4, 5]
    assert dataset.predict_df().collect().get_column("id").to_list() == [0, 1, 2, 3, 4, 5]

    assert dataset.train_class_weights() == {0: 0.75, 1: 1.5}
    assert dataset.val_class_weights() == {0: 1.5, 1: 0.75}
    assert dataset.test_class_weights() == {0: 1.0, 1: 1.0}
    assert dataset.predict_class_weights() == {0: 1.0, 1: 1.0}

    np.testing.assert_allclose(
        np.sort(dataset.train_sample_weights()), np.array([0.75, 0.75, 1.5])
    )
    np.testing.assert_allclose(
        np.sort(dataset.val_sample_weights()), np.array([0.75, 0.75, 1.5])
    )
    np.testing.assert_allclose(dataset.test_sample_weights(), np.ones(6))
    np.testing.assert_allclose(dataset.predict_sample_weights(), np.ones(6))

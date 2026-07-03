import polars as pl
import numpy as np
from pathlib import Path
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralnet.workflows.mlp.dataset import RingerParquetDataset


def test_training_job(
    tmp_path: Path,
    ringer_dataset_dir: Path,
    ringer_parquet_dataset: "RingerParquetDataset",
    isolated_executor,
):
    future = isolated_executor.submit(
        training_job_test_routine,
        tmp_path=tmp_path,
        dataset_dir=ringer_dataset_dir,
        ringer_parquet_dataset=ringer_parquet_dataset,
    )
    future.result()


def training_job_test_routine(
    tmp_path: Path, dataset_dir: Path, ringer_parquet_dataset: "RingerParquetDataset"
):

    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    from neuralnet.workflows.mlp.training import (
        MLPKerasTrainingJob,
    )
    import pandera.polars as pa
    from neuralnet.submitit import ExecutorConfig

    job_config = {
        "batch_size": 32,
        "data_table": "data",
        "et_col": "et",
        "et_bins": [{"low": 0.0, "high": 100000.0, "closed": "left"}],
        "eta_col": "eta",
        "eta_bins": [{"low": 0.0, "high": 2.5, "closed": "left"}],
        "fold_col": "fold",
        "kfold_table": "kfold",
        "label_col": "label",
        "norm_strategy": "l1",
        "rings_col": "rings",
        "ring_fraction": 2,
        "model_factory": {
            "layers": [
                {
                    "units": 4,
                    "activation": "relu",
                    "name": "hidden_dense",
                },
                {
                    "units": 1,
                    "activation": "sigmoid",
                    "name": "output_dense",
                },
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
        "inits": 2,
    }

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
    assert loaded_job.all_model_results.height == n_folds * job.inits * len(
        job.et_bins
    ) * len(job.eta_bins)
    assert loaded_job.selected_models.height == len(job.et_bins) * len(job.eta_bins)

    inference_pipeline = loaded_job.get_inference_pipeline()
    inference_input = ringer_parquet_dataset.predict_df()
    inference_results = inference_pipeline(
        inference_input, join_results=True, all_layers=True
    ).collect()

    assert isinstance(inference_results, pl.DataFrame), (
        "Inference results is not a Polars DataFrame"
    )
    assert inference_results.height == inference_input.collect().height

    inference_results_expected_schema = {
        "id": pa.Column(pl.UInt64, nullable=False),
        "et": pa.Column(pl.Float32, nullable=False),
        "eta": pa.Column(pl.Float32, nullable=False),
        "fold": pa.Column(pl.Int64, nullable=False),
        "label": pa.Column(pl.Boolean, nullable=False),
        "rings": pa.Column(pl.List(pl.Float32), nullable=False),
        "output": pa.Column(pl.Float32, nullable=False),
        "prediction": pa.Column(pl.Boolean, nullable=False),
    }
    for layer_config in job_config["model_factory"]["layers"]:
        for i in range(layer_config["units"]):
            inference_results_expected_schema[f"layer.{layer_config['name']}.{i}"] = pa.Column(
                pl.Float32, nullable=False
            )

    inference_results_expected_schema = pa.DataFrameSchema(
        inference_results_expected_schema
    )

    inference_results_expected_schema.validate(inference_results, lazy=False)

    logging.info("Finished")


def test_ringer_parquet_dataset_splits_and_weights(
    small_ringer_parquet_dataset: "RingerParquetDataset",
):
    dataset = small_ringer_parquet_dataset

    assert dataset.get_n_folds() == 2

    rings_expr, ring_names = dataset.open_rings_expr()
    assert len(rings_expr) == dataset.N_RINGS
    assert len(ring_names) == dataset.N_RINGS
    assert ring_names[0] == "rings.0"
    assert ring_names[-1] == "rings.99"

    assert dataset.train_df().collect().get_column("id").to_list() == [2, 3, 5]
    assert dataset.val_df().collect().get_column("id").to_list() == [0, 1, 4]
    assert dataset.test_df().collect().get_column("id").to_list() == [0, 1, 2, 3, 4, 5]
    assert dataset.predict_df().collect().get_column("id").to_list() == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]

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

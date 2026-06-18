import pytest
import polars as pl
import numpy as np
from pathlib import Path
import logging


MLP_JOB_CONFIG = {
    "batch_size": 32,
    "data_table": "data",
    "et_col": "et",
    "et_bins": [0, 100],
    "eta_col": "eta",
    "eta_bins": [0, 2.5],
    "fold_col": "fold",
    "kfold_table": "kfold",
    "label_col": "target_label",
    "rings_col": "rings",
    "ring_fraction": 1,
    "model_factory": {
        "object_type": "keras_sequential",
        "layers": [
            {"object_type": "dense", "units": 4, "activation": "relu"},
            {"object_type": "dense", "units": 1, "activation": "sigmoid"},
        ],
        "name": "mlp"
    },
    "loss": {"object_type": "binary_cross_entropy", "from_logits": False},
    "optimizer": {"learning_rate": 0.01, "object_type": "adam"},
    "from_logits": False,
    "num_thresholds": 10,
    "lower_threshold": 0.1,
    "upper_threshold": 0.9,
    "epochs": 1,
    "logger_name": None,
    "output_path": None,  # To be set in the test
    "executor_config": None,  # To be set in the test
    "inits": 1,
}

TEST_DATA = {
    "mlp": MLP_JOB_CONFIG,
}


@pytest.mark.parametrize(
    ("job_config"),
    list(TEST_DATA.values()),
    ids=list(TEST_DATA.keys()),
)
def test_binary_classification_job(tmp_path: Path, job_config: dict):
    from neuralnet.ringer_committee_trigger.keras_training_job import (
        RingerCommitteeKerasTrainingJob,
    )
    from neuralnet.submitit import ExecutorConfig
    # 1. Create a random dataset
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    job_config["dataset_dir"] = dataset_dir

    n_samples = 100
    n_rings = 100

    # Create data_table
    data_df = pl.DataFrame(
        {
            "id": np.arange(n_samples),
            "rings": [np.random.rand(n_rings).tolist() for _ in range(n_samples)],
            "et": np.random.rand(n_samples) * 100.0,
            "eta": np.random.rand(n_samples) * 2.5,
        }
    )
    data_df.write_parquet(dataset_dir / "data.parquet")

    # Create kfold_table
    kfold_df = pl.DataFrame(
        {
            "id": np.arange(n_samples),
            "target_label": np.random.randint(0, 2, size=n_samples),
            "fold": [i % 2 for i in range(n_samples)],  # 2 folds
        }
    )
    kfold_df.write_parquet(dataset_dir / "kfold.parquet")

    # 4. Configure Executor
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

    job = RingerCommitteeKerasTrainingJob(**job_config)
    # 6. Run the job
    job.submit()

    loaded_job = RingerCommitteeKerasTrainingJob.load(job_config["output_path"])
    logging.info("Finished")

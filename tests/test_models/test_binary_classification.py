import pytest
import polars as pl
import numpy as np
from pathlib import Path
from zipfile import ZipFile

from neuralnet.datasets.ringer import RingerParquetDataset, Bin
from neuralnet.models.binary_classification import (
    BinaryClassificationJob,
    BinaryClassificationModel,
)
from neuralnet.models.mlp import DenseLayer
from neuralnet.models.quantum import (
    BasicEntanglerQuantumLayer,
    StronglyEntanglingQuantumLayer,
    HardwareEfficientQuantumLayer,
)
from neuralnet.optimizers import AdamOptimizer
from neuralnet.losses import BinaryCrossEntropyLossConfig
from neuralnet.submitit import ExecutorConfig


def test_binary_classification_job(tmp_path: Path):
    # 1. Create a random dataset
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

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

    # 2. Configure dataset
    dataset_config = RingerParquetDataset(
        dataset_dir=dataset_dir,
        data_table="data",
        rings_col="rings",
        kfold_table="kfold",
        label_col="target_label",
        fold_col="fold",
        et_col="et",
        et_bin=Bin(low=0.0, high=100.0, closed="left"),
        eta_col="eta",
        eta_bin=Bin(low=0.0, high=2.5, closed="left"),
        ring_fraction=1,
        batch_size=32,
        kind="ringer_dataset",
    )

    # 3. Configure Model
    model_config = BinaryClassificationModel(
        name="test_model",
        layers=[
            DenseLayer(units=4, activation="relu"),
            DenseLayer(units=1, activation="sigmoid"),
        ],
        loss=BinaryCrossEntropyLossConfig(
            kind="binary_cross_entropy", from_logits=False
        ),
        optimizer=AdamOptimizer(learning_rate=0.01, kind="adam"),
        from_logits=False,
        num_thresholds=10,
        lower_threshold=0.1,
        upper_threshold=0.9,
        epochs=1,
        logger_name="test_logger",
    )

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

    # 5. Configure Job
    output_path = tmp_path / "output"
    job = BinaryClassificationJob(
        dataset=dataset_config,
        model=model_config,
        executor_config=executor_config,
        inits=1,
        output_path=output_path,
        logger_name="test_logger",
    )

    # 6. Run the job
    job.run()

    # 7. Check output
    assert output_path.exists()
    zip_0 = output_path / "fold_0_init_0.zip"
    zip_1 = output_path / "fold_1_init_0.zip"

    assert zip_0.exists()
    assert zip_1.exists()

    job_config_path = output_path / "job_config.json"
    assert job_config_path.exists()

    import json

    # Check contents of zip_0
    with ZipFile(zip_0, mode="r") as archive:
        files = archive.namelist()
        assert "model/model_config.json" in files
        assert "model/keras_model.keras" in files
        assert "results.json" in files
        assert "config.json" in files

        # Verify results.json content
        with archive.open("results.json") as f:
            results = json.load(f)
            assert "fit" in results
            assert "train" in results
            assert "val" in results
            assert "test" in results

        # Verify config.json content
        with archive.open("config.json") as f:
            config = json.load(f)
            assert config["name"] == "test_model"

    # Verify model is correctly saved and can be loaded
    loaded_model = BinaryClassificationModel.load(zip_0, base_dir="model/")
    assert loaded_model.name == "test_model"
    assert loaded_model.keras is not None

    loaded_job = BinaryClassificationJob.load(output_path)
    assert loaded_job.model.name == "test_model"
    assert len(loaded_job.models) == 2
    assert all(model.keras is not None for model in loaded_job.models)

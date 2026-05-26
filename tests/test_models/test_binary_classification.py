import pytest
import polars as pl
import numpy as np
from pathlib import Path
from keras.models import Sequential

from neuralnet.models.binary_classification import (
    BinaryClassificationJob,
    BinaryClassificationModel,
)
from neuralnet.models.mlp import DenseLayer
from neuralnet.optimizers import AdamOptimizer
from neuralnet.losses import BinaryCrossEntropyLossConfig
from neuralnet.submitit import ExecutorConfig


DATASET_CONFIG = {
    "data_table": "data",
    "rings_col": "rings",
    "kfold_table": "kfold",
    "label_col": "target_label",
    "fold_col": "fold",
    "et_col": "et",
    "et_bin": {"low": 0.0, "high": 100.0, "closed": "left"},
    "eta_col": "eta",
    "eta_bin": {"low": 0.0, "high": 2.5, "closed": "left"},
    "ring_fraction": 1,
    "batch_size": 32,
    "kind": "ringer_dataset",
}

JOB_CONFIG = {
    "dataset": None,  # To be set in the test
    "model": None,  # To be set in the test
    "executor_config": None,  # To be set in the test
    "inits": 1,
    "output_path": None,  # To be set in the test
    "logger_name": None,
}


MLP_CONFIG = {
    "name": "test_model",
    "layers": [
        {"name": "dense", "units": 4, "activation": "relu"},
        {"name": "dense", "units": 1, "activation": "sigmoid"},
    ],
    "loss": {"kind": "binary_cross_entropy", "from_logits": False},
    "optimizer": {"learning_rate": 0.01, "kind": "adam"},
    "from_logits": False,
    "num_thresholds": 10,
    "lower_threshold": 0.1,
    "upper_threshold": 0.9,
    "epochs": 1,
    "logger_name": None,
}

BASIC_ENTANGLER_CONFIG = {
    "name": "test_quantum_model_basic_entangler",
    "layers": [
        {"name": "dense", "units": 4, "activation": "relu"},
        {"name": "basic_entangler", "n_qubits": 4, "n_layers": "2"},
        {"name": "dense", "units": 1, "activation": "sigmoid"},
    ],
    "loss": {"kind": "binary_cross_entropy", "from_logits": False},
    "optimizer": {"learning_rate": 0.01, "kind": "adam"},
    "from_logits": False,
    "num_thresholds": 10,
    "lower_threshold": 0.1,
    "upper_threshold": 0.9,
    "epochs": 1,
    "logger_name": None,
}

STRONGLY_ENTANGLING_CONFIG = {
    "name": "test_quantum_model_strongly_entangling",
    "layers": [
        {"name": "dense", "units": 4, "activation": "relu"},
        {"name": "strongly_entangling", "n_qubits": 4, "n_layers": "2"},
        {"name": "dense", "units": 1, "activation": "sigmoid"},
    ],
    "loss": {"kind": "binary_cross_entropy", "from_logits": False},
    "optimizer": {"learning_rate": 0.01, "kind": "adam"},
    "from_logits": False,
    "num_thresholds": 10,
    "lower_threshold": 0.1,
    "upper_threshold": 0.9,
    "epochs": 1,
    "logger_name": None,
}

HARDWARE_EFFICIENT_CONFIG = {
    "name": "test_quantum_model_hardware_efficient",
    "layers": [
        {"name": "dense", "units": 4, "activation": "relu"},
        {"name": "hardware_efficient", "n_qubits": 4, "n_layers": "2"},
        {"name": "dense", "units": 1, "activation": "sigmoid"},
    ],
    "loss": {"kind": "binary_cross_entropy", "from_logits": False},
    "optimizer": {"learning_rate": 0.01, "kind": "adam"},
    "from_logits": False,
    "num_thresholds": 10,
    "lower_threshold": 0.1,
    "upper_threshold": 0.9,
    "epochs": 1,
    "logger_name": None,
}


TEST_DATA = {
    "mlp": (DATASET_CONFIG, JOB_CONFIG, MLP_CONFIG),
    "basic_entangler": (DATASET_CONFIG, JOB_CONFIG, BASIC_ENTANGLER_CONFIG),
    "strongly_entangling": (DATASET_CONFIG, JOB_CONFIG, STRONGLY_ENTANGLING_CONFIG),
    "hardware_efficient": (DATASET_CONFIG, JOB_CONFIG, HARDWARE_EFFICIENT_CONFIG),
}


@pytest.mark.parametrize(
    ("dataset_config", "job_config", "model_config"),
    list(TEST_DATA.values()),
    ids=list(TEST_DATA.keys()),
)
def test_binary_classification_job(
    tmp_path: Path, dataset_config: dict, job_config: dict, model_config: dict
):
    # 1. Create a random dataset
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset_config["dataset_dir"] = dataset_dir
    job_config["dataset"] = dataset_config
    job_config["model"] = model_config

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

    job = BinaryClassificationJob(**job_config)
    # 6. Run the job
    job.submit()

    loaded_job = BinaryClassificationJob.load(job_config["output_path"])
    assert loaded_job.model.name == model_config["name"]
    assert len(loaded_job.models) == 2
    for i, model in enumerate(loaded_job.models):
        assert isinstance(model, BinaryClassificationModel), (
            f"Model {i} is not an instance of BinaryClassificationModel"
        )
        assert isinstance(model.keras, Sequential), (
            f"Keras model in model {i} is not an instance of BinaryClassificationModel"
        )


def test_binary_classification_model_save_load_predict_same_output(tmp_path: Path):
    model = BinaryClassificationModel(
        name="predict_equivalence_model",
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

    rng = np.random.default_rng(42)
    x = rng.normal(size=(8, 100)).astype(np.float32)

    original_predictions = model.predict(x)

    model_path = tmp_path / "binary_model.zip"
    model.save(model_path)
    loaded_model = BinaryClassificationModel.load(model_path)

    loaded_predictions = loaded_model.predict(x)

    np.testing.assert_allclose(
        original_predictions, loaded_predictions, rtol=1e-6, atol=1e-7
    )

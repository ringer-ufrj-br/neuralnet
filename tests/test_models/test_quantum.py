import pytest
from zipfile import ZipFile
from pathlib import Path
import polars as pl
import numpy as np
import keras

from neuralnet.models import quantum
from neuralnet.models.mlp import DenseLayer
from neuralnet.models.binary_classification import (
    BinaryClassificationModel,
    BinaryClassificationJob,
)
from neuralnet.datasets.ringer import RingerParquetDataset, Bin
from neuralnet.losses import BinaryCrossEntropyLossConfig
from neuralnet.submitit import ExecutorConfig


def _run_quantum_layer_factory(
    class_name: str, kwargs: dict, name: str, monkeypatch: pytest.MonkeyPatch
):
    config_cls = getattr(quantum, class_name)
    config: quantum.QuantumLayer = config_cls(**kwargs)

    assert config.diff_method == kwargs.get("diff_method", "backprop")
    assert config.shots == kwargs.get("shots")
    assert config.name == name

    layer = config.get()

    assert isinstance(layer, keras.layers.TorchModuleWrapper), (
        "Expected instance of 'TorchModuleWrapper', got '{}'".format(
            type(layer).__name__
        )
    )
    assert str(layer.output_shape) == str(
        kwargs.get("output_shape", (None, kwargs["n_qubits"]))
    )
    assert layer.name == kwargs["name"]


@pytest.mark.parametrize(
    ("class_name", "kwargs", "name"),
    [
        (
            "BasicEntanglerQuantumLayer",
            {
                "n_qubits": 2,
                "n_layers": 2,
                "name": "basic_entangler",
                "shots": None,
                "diff_method": "backprop",
            },
            "basic_entangler",
        ),
        (
            "StronglyEntanglingQuantumLayer",
            {
                "n_qubits": 3,
                "n_layers": 1,
                "name": "strongly_entangling",
                "shots": 32,
                "diff_method": "parameter-shift",
            },
            "strongly_entangling",
        ),
        (
            "HardwareEfficientQuantumLayer",
            {
                "n_qubits": 4,
                "n_layers": 2,
                "name": "hardware_efficient",
                "output_shape": (None, 7),
                "shots": 128,
                "diff_method": "finite-diff",
            },
            "hardware_efficient",
        ),
    ],
)
def test_quantum_layer_config_get_returns_torch_module_wrapper(
    class_name: str, kwargs: dict, name: str, monkeypatch: pytest.MonkeyPatch
):
    _run_quantum_layer_factory(class_name, kwargs, name, monkeypatch)


@pytest.mark.parametrize(
    "quantum_layer_class_name, name",
    [
        ("BasicEntanglerQuantumLayer", "basic_entangler"),
        ("StronglyEntanglingQuantumLayer", "strongly_entangling"),
        ("HardwareEfficientQuantumLayer", "hardware_efficient"),
    ],
)
def test_binary_classification_job_quantum(
    tmp_path: Path, quantum_layer_class_name, name, monkeypatch
):

    quantum_layer_class = getattr(quantum, quantum_layer_class_name)

    # Create a random dataset
    dataset_dir = tmp_path / f"dataset_{name}"
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
        name=f"test_quantum_model_{name}",
        layers=[
            DenseLayer(units=4, activation="relu"),
            quantum_layer_class(n_qubits=4, n_layers=2),
            DenseLayer(units=1, activation="sigmoid"),
        ],
        loss=BinaryCrossEntropyLossConfig(
            kind="binary_cross_entropy", from_logits=False
        ),
        optimizer={"learning_rate": 0.01, "kind": "adam"},
        from_logits=False,
        num_thresholds=10,
        lower_threshold=0.1,
        upper_threshold=0.9,
        epochs=1,
        logger_name="test_logger",
    )

    # 4. Configure Executor
    logs_dir = tmp_path / f"logs_{name}"
    logs_dir.mkdir()
    executor_config = ExecutorConfig(
        cpus_per_task=1,
        executor_type="debug",
        logs_dir=logs_dir,
        name=f"test_job_quantum_{name}",
        slurm_partition="test",
    )

    # 5. Configure Job
    output_path = tmp_path / f"output_quantum_{name}"
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
            assert config["name"] == f"test_quantum_model_{name}"
            assert len(config["layers"]) == 3
            assert config["layers"][1]["name"] == name

import polars as pl
import numpy as np
from pathlib import Path
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralnet.datasets.ringer import RingerParquetDataset


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

    from neuralnet.workflows.ringer.training import (
        RingerKerasTrainingJob,
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
            "object_type": "mlp",
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
        "best_init": {
            "key": "fit.val.max_sp",
            "mode": "max",
        },
        "best_fold": {
            "key": "fit.val.max_sp",
            "mode": "max",
        }
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

    job = RingerKerasTrainingJob(**job_config)
    job.submit()

    loaded_job = RingerKerasTrainingJob.load(job_config["output_path"])
    assert isinstance(loaded_job, RingerKerasTrainingJob), (
        "Loaded job is not of the expected type"
    )
    assert loaded_job == job, "Loaded job is not equal to the original job"
    RingerKerasTrainingJob.validate_saved_directory(job_config["output_path"])
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

    specialist_committee = loaded_job.get_specialist_committee()
    inference_input = ringer_parquet_dataset.predict_df()
    inference_results = specialist_committee.predict(
        inference_input, passthrough=True
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
        # "prediction": pa.Column(pl.Boolean, nullable=False),
    }
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


# ---------------------------------------------------------------------------
# Threshold fit job tests
# ---------------------------------------------------------------------------


def test_threshold_fit_job(
    tmp_path: Path,
    ringer_dataset_dir: Path,
    ringer_parquet_dataset: "RingerParquetDataset",
    isolated_executor,
):
    """End-to-end test for RingerCommitteeThresholdFitJob.

    First trains a minimal committee (via the existing training routine) then
    runs the evaluation job against the same dataset directory.  Asserts that
    all expected output files are created with valid contents.
    """
    future = isolated_executor.submit(
        threshold_fit_job_test_routine,
        tmp_path=tmp_path,
        dataset_dir=ringer_dataset_dir,
        ringer_parquet_dataset=ringer_parquet_dataset,
    )
    future.result()


def threshold_fit_job_test_routine(
    tmp_path: Path,
    dataset_dir: Path,
    ringer_parquet_dataset: "RingerParquetDataset",
) -> None:
    """Helper that runs inside an isolated process to avoid Keras import side-effects."""

    import os
    import json

    os.environ["KERAS_BACKEND"] = "tensorflow"

    from neuralnet.workflows.ringer.training import RingerKerasTrainingJob
    from neuralnet.workflows.ringer.threshold_fit import (
        RingerCommitteeThresholdFitJob,
        ReferencePoint,
    )
    from neuralnet.submitit import ExecutorConfig

    # ------------------------------------------------------------------
    # Step 1: Train a minimal committee.
    # ------------------------------------------------------------------
    training_output = tmp_path / "training_output"
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    executor_config = ExecutorConfig(
        cpus_per_task=1,
        executor_type="debug",
        logs_dir=logs_dir,
        name="test_eval_training_job",
        slurm_partition="test",
    )

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
            "object_type": "mlp",
            "layers": [
                {"units": 4, "activation": "relu", "name": "hidden_dense"},
                {"units": 1, "activation": "sigmoid", "name": "output_dense"},
            ],
            "name": "mlp",
        },
        "loss": {"object_type": "binary_cross_entropy", "from_logits": False},
        "optimizer": {"learning_rate": 0.01, "object_type": "adam"},
        "from_logits": False,
        "epochs": 1,
        "logger_name": None,
        "output_path": training_output,
        "executor_config": executor_config,
        "dataset_dir": dataset_dir,
        "inits": 1,
        "best_init": {"key": "fit.val.max_sp", "mode": "max"},
        "best_fold": {"key": "fit.val.max_sp", "mode": "max"},
    }

    training_job = RingerKerasTrainingJob(**job_config)
    training_job.submit()

    # Sanity check — training must have succeeded.
    RingerKerasTrainingJob.validate_saved_directory(training_output)

    # ------------------------------------------------------------------
    # Step 2: Run the evaluation job.
    # ------------------------------------------------------------------
    evaluation_output = tmp_path / "evaluation_output"

    references = {
        "tight": ReferencePoint(tpr=0.995, color="red", label="Tight"),
        "medium": ReferencePoint(tpr=0.996, color="orange", label="Medium"),
    }

    eval_job = RingerCommitteeThresholdFitJob(
        fit_job_path=training_output,
        dataset_dir=dataset_dir,
        output_path=evaluation_output,
        references=references,
        dataset_types=["train", "val", "test"],
        batch_size=32,
    )
    eval_job.submit()

    # ------------------------------------------------------------------
    # Step 3: Validate outputs.
    # ------------------------------------------------------------------

    # results.parquet must exist and have one row per selected specialist.
    assert eval_job.results_path.exists(), (
        f"results.parquet not found at {eval_job.results_path}"
    )
    results_df = pl.read_parquet(eval_job.results_path)
    assert isinstance(results_df, pl.DataFrame), "results.parquet is not a DataFrame"

    loaded_training_job = RingerKerasTrainingJob.load(training_output)
    n_selected = loaded_training_job.selected_models.height
    assert results_df.height == n_selected, (
        f"Expected {n_selected} rows in results.parquet, got {results_df.height}"
    )

    # committee_results.json must exist and be valid JSON.
    assert eval_job.committee_results_path.exists(), (
        f"committee_results.json not found at {eval_job.committee_results_path}"
    )
    with eval_job.committee_results_path.open("r", encoding="utf-8") as fh:
        committee_results = json.load(fh)
    assert isinstance(committee_results, dict), (
        "committee_results.json does not contain a dict"
    )

    # Check that expected keys exist in committee_results for every
    # (dataset_type, reference, metric) combination.
    expected_metrics = ["tpr", "fpr", "accuracy", "sp"]
    for dataset_type, ref_name in [
        (dt, rn)
        for dt in eval_job.dataset_types
        for rn in eval_job.references
    ]:
        for metric in expected_metrics:
            key = f"{dataset_type}.{ref_name}.{metric}"
            assert key in committee_results, (
                f"Expected key '{key}' missing from committee_results.json"
            )

    # Each member directory must contain confusion matrix parquet files.
    member_dirs = list(evaluation_output.glob("member_*"))
    assert len(member_dirs) == n_selected, (
        f"Expected {n_selected} member directories, found {len(member_dirs)}"
    )
    for member_dir in member_dirs:
        for dataset_type in eval_job.dataset_types:
            cm_path = member_dir / f"{dataset_type}_confusion_matrix.parquet"
            assert cm_path.exists(), (
                f"Confusion matrix parquet file missing: {cm_path}"
            )
            cm_df = pl.read_parquet(cm_path)
            assert isinstance(cm_df, pl.DataFrame)
            assert "tpr" in cm_df.columns
            assert "fpr" in cm_df.columns
            assert "threshold" in cm_df.columns


    # models/ directory must exist and contain the member model files.
    assert eval_job.models_dir.exists(), f"models directory not found at {eval_job.models_dir}"
    model_files = list(eval_job.models_dir.glob("*.keras"))
    assert len(model_files) == n_selected, (
        f"Expected {n_selected} model files in {eval_job.models_dir}, found {len(model_files)}"
    )

    # Operating-point JSON files must exist for each reference.
    from neuralnet.workflows.ringer.models import BinnedSpecialistCommittee

    for ref_name in eval_job.references:
        op_json_path = eval_job.get_op_point_json_path(ref_name)
        assert op_json_path.exists(), f"Operating point JSON not found at {op_json_path}"
        with op_json_path.open("r", encoding="utf-8") as fh:
            op_data = json.load(fh)
        assert op_data["op_point"] == ref_name
        assert "reference" in op_data
        assert "preprocessing" in op_data
        assert "models" in op_data
        assert len(op_data["models"]) == n_selected
        for model_cfg in op_data["models"]:
            assert "model_path" in model_cfg
            assert "decision_threshold" in model_cfg
            assert "bins" in model_cfg
            assert "preprocessing" in model_cfg

    # Test loading committee via RingerCommitteeThresholdFitJob.get_specialist_committee
    committee = RingerCommitteeThresholdFitJob.get_specialist_committee(
        path=evaluation_output,
        op_point="tight",
        dataset_type="test",
    )
    assert isinstance(committee, BinnedSpecialistCommittee)
    assert len(committee.models) == n_selected, (
        f"get_specialist_committee returned {len(committee.models)} models, expected {n_selected}"
    )
    for specialist in committee.models:
        assert specialist.decision_threshold is not None, (
            f"decision_threshold not set for specialist {specialist.training_results.get('id')}"
        )
        assert isinstance(specialist.decision_threshold, float), (
            "decision_threshold must be a float"
        )
  
    # Test loading committee directly via BinnedSpecialistCommittee.from_json
    committee_from_json = BinnedSpecialistCommittee.from_json(
        evaluation_output / "tight.json"
    )
    assert isinstance(committee_from_json, BinnedSpecialistCommittee)
    assert len(committee_from_json.models) == n_selected

    # Test prediction on a sample dataset
    sample_dataset = loaded_training_job.get_dataset(fold=0)
    pred_df = sample_dataset.predict_df().pipe(committee.predict, passthrough=True)
    if isinstance(pred_df, pl.LazyFrame):
        pred_df = pred_df.collect()
    assert "prediction" in pred_df.columns, "'prediction' column missing in committee output"
    assert "output" in pred_df.columns, "'output' column missing in committee output"
    assert pred_df["prediction"].dtype == pl.Boolean

    logging.info("Threshold fit job test passed successfully.")


from pathlib import Path


def test_vqat_pipeline_from_yaml_routine(test_data_dir: Path, isolated_executor):
    future = isolated_executor.submit(
        vqat_pipeline_from_yaml_routine,
        test_data_dir=test_data_dir,
    )
    future.result()


def vqat_pipeline_from_yaml_routine(test_data_dir: Path):
    import shutil
    import polars as pl
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"

    from neuralnet.datasets import ParquetDataset
    from neuralnet.models.vqat import VQATTrainingJob, add_inference

    dataset_dir = test_data_dir / "test_dataset"
    training_dir = dataset_dir / "training" / "vqat"
    inference_dir = dataset_dir / "inference" / "vqat"
    if training_dir.exists():
        shutil.rmtree(training_dir)
    if inference_dir.exists():
        shutil.rmtree(inference_dir)
    training_dir.mkdir(parents=True, exist_ok=True)

    job = VQATTrainingJob.from_yaml(
        test_data_dir / "vqat_training_job.yaml",
        output_dir=training_dir,
        dataset_dir=dataset_dir,
    )
    job.run()

    add_inference(
        results_dir=training_dir,
        dataset_dir=dataset_dir,
        features_table="electron_ringer",
        inference_table="inference/vqat_inference_results",
    )

    dataset = ParquetDataset(dataset_dir=dataset_dir)
    inference_df = pl.read_parquet(
        dataset.get_table_path("inference/vqat_inference_results")
    )

    assert "prediction" in inference_df.columns
    assert inference_df.height > 0

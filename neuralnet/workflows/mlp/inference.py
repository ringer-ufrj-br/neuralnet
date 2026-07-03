from functools import cached_property
from pydantic import Field
from pathlib import Path
import logging
from typing import Annotated, Self
import json
import polars as pl

from ...submitit import ExecutorConfig
from .jobs import MLPKerasTrainingJob
from .dataset import (
    DataTableType,
    RingsColType,
    EtColType,
    EtaColType,
)
from ...datasets import DirectoryType, ParquetDataset
from ...pydantic import YamlBaseModel
from ...quantization.hgq import HGQFixedPointConfig


class InferenceJob(YamlBaseModel):
    # job Related fields
    fit_job_path: Annotated[
        Path,
        Field(
            description="Path to the outut path from a MLPKerasTrainingJob. This is used to load the selected models for inference.",
        ),
    ]

    # Dataset related fields
    data_table: DataTableType
    dataset_dir: DirectoryType
    et_col: EtColType
    eta_col: EtaColType
    rings_col: RingsColType
    fold_col: str | None = None
    kfold_table: str | None = None
    batch_size: Annotated[
        int,
        Field(
            description="Size of the batches to use for inference.",
        ),
    ] = 32

    # Output related fields
    output_table: Annotated[
        str,
        Field(
            description="Name of the output table where the predictions will be stored.",
        ),
    ]
    max_rows_per_file: Annotated[
        int,
        Field(
            default=100_000,
            description="Maximum number of rows to write for each output file.",
        ),
    ]

    def submit(self) -> pl.LazyFrame:
        logger = logging.getLogger()
        logger.info(f"Starting inference job with fit_job_path {self.fit_job_path}")
        training_job = MLPKerasTrainingJob.load(self.fit_job_path)
        dataset = ParquetDataset(dataset_dir=self.dataset_dir)
        data_table = dataset.get_dataframe(self.data_table)

        if self.fold_col and self.kfold_table:
            kfold_df = dataset.get_dataframe(self.kfold_table)
            data_table = data_table.join(kfold_df, on="id", how="left")

        prediction_function, _, _ = training_job.get_committee_model(
            et_col=self.et_col,
            eta_col=self.eta_col,
            rings_col=self.rings_col,
            fold_col=self.fold_col if self.fold_col else False,
        )
        logger.info(f"Loaded model, running inference on data table {self.data_table}")
        prediction_df = prediction_function(data_table, self.batch_size)
        output_table_path = dataset.get_table_path(self.output_table)
        logger.info(f"Writing predictions to {output_table_path}")
        prediction_df.write_parquet(
            pl.PartitionBy(
                str(output_table_path),
                max_rows_per_file=self.max_rows_per_file,
            )
        )
        return prediction_df


class PTQConversionJob(InferenceJob):
    # Quantization related fields
    weight_quantization: Annotated[
        HGQFixedPointConfig,
        Field(
            description="Configuration for the weight quantization.",
        ),
    ]
    bias_quantization: Annotated[
        HGQFixedPointConfig,
        Field(
            description="Configuration for the bias quantization.",
        ),
    ]
    output_path: Annotated[
        Path,
        Field(
            description="Path to the output directory where the results of the inference job will be stored.",
        ),
    ]

    executor_config: Annotated[
        ExecutorConfig,
        Field(
            description="Slurm configuration for running the training job on a Slurm cluster"
        ),
    ]

    @cached_property
    def all_models_results_path(self) -> Path:
        return self.output_path / "all_models_results.parquet"
    
    @cached_property
    def all_model_results(self) -> pl.DataFrame:
        results_path = self.output_path / "all_models_results.parquet"
        if not results_path.exists():
            raise FileNotFoundError(
                f"All models results file not found at {results_path}"
            )
        return pl.read_parquet(results_path)

    @cached_property
    def selected_models_path(self) -> Path:
        return self.output_path / "selected_models.parquet"

    @cached_property
    def selected_models(self) -> pl.DataFrame:
        selected_models_path = self.output_path / "selected_models.parquet"
        if not selected_models_path.exists():
            raise FileNotFoundError(
                f"Selected models file not found at {selected_models_path}"
            )
        return pl.read_parquet(selected_models_path)

    def submit(self) -> pl.LazyFrame:
        logger = logging.getLogger()
        logger.info(
            f"Starting uniform PTQ inference job with fit_job_path {self.fit_job_path}"
        )

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_path.joinpath("config.json").write_text(
            self.model_dump_json(indent=4),
            encoding="utf-8",
        )
        executor = self.executor_config.get_executor()
        submitted_jobs = []
        training_job = MLPKerasTrainingJob.load(self.fit_job_path)
        with executor.batch():
            length = len(training_job.selected_models)
            for i, model_result in enumerate(
                training_job.selected_models.select("id").iter_rows(named=True)
            ):
                logger.info(
                    f"Submitting job for model {model_result['id']} ({i + 1}/{length})"
                )
                submitted_job = executor.submit(
                    self.run_evaluation,
                    member_id=model_result["id"],
                    submission_id=i,
                )
                submitted_jobs.append(submitted_job)

        dependent_executor = self.executor_config.get_executor()
        from submitit import AutoExecutor

        if isinstance(dependent_executor, AutoExecutor) and submitted_jobs:
            dependency_string = ":".join(str(job.job_id) for job in submitted_jobs)
            logger.info(
                f"Submitting dependent job with dependency on jobs: {dependency_string}"
            )
            dependent_executor.update_parameters(
                slurm_additional_parameters={
                    "dependency": f"afterok:{dependency_string}"
                }
            )
        logger.info("Submitting final evaluation job to aggregate results.")
        dependent_executor.submit(self.final_evaluation)

        logger.info("All jobs submitted.")

    def run_evaluation(self, member_id: int, submission_id: int | None = None) -> None:
        from ...quantization.hgq import hgq_quantize

        logger = logging.getLogger()
        logger.info(
            f"{submission_id}:Running evaluation for PTQ conversion job with fit_job_path {self.fit_job_path}"
        )
        training_job = MLPKerasTrainingJob.load(self.fit_job_path)
        member_output_path = self.output_path / f"member_{member_id}"
        # if member_output_path.exists():
        #     return
        member_output_path.mkdir(parents=True, exist_ok=True)
        model, results = training_job.get_member_model(member_id, with_results=True)
        quantized_model_result = results.copy()
        quantized_model_result["weight_quantization"] = (
            self.weight_quantization.model_dump()
        )
        quantized_model_result["bias_quantization"] = (
            self.bias_quantization.model_dump()
        )
        quantized_model = hgq_quantize(
            model,
            weight_quantizer_config=self.weight_quantization,
            bias_quantizer_config=self.bias_quantization,
        )
        # quantized_model = model
        dataset = training_job.get_dataset(
            fold=quantized_model_result["fold"],
            et_bin=quantized_model_result["et_bin"],
            eta_bin=quantized_model_result["eta_bin"],
        )
        quantized_model_result["train"] = training_job.run_evaluation(
            model=quantized_model,
            data=dataset.train_numpy(),
            class_weight=dataset.train_class_weights()
            if training_job.balance_class_weights
            else None,
        )
        quantized_model_result["val"] = training_job.run_evaluation(
            model=quantized_model,
            data=dataset.val_numpy(),
            class_weight=dataset.val_class_weights()
            if training_job.balance_class_weights
            else None,
        )
        quantized_model_result["test"] = training_job.run_evaluation(
            model=quantized_model,
            data=dataset.test_numpy(),
            class_weight=dataset.test_class_weights()
            if training_job.balance_class_weights
            else None,
        )
        model_path = training_job.get_member_model_path(member_output_path)
        quantized_model.save(model_path)
        from ...json import cast_to_json_value

        with member_output_path.joinpath("results.json").open("w") as f:
            json.dump(cast_to_json_value(quantized_model_result), f, indent=4)

        logger.info(
            f"{submission_id}:Finished evaluation for PTQ conversion job with fit_job_path {self.fit_job_path} and member_id {member_id}"
        )

    def final_evaluation(self) -> None:
        from collections import defaultdict
        from ...utils import traverse

        logger = logging.getLogger()
        training_job = MLPKerasTrainingJob.load(self.fit_job_path)
        all_models_results = defaultdict(list)
        for member_path in self.output_path.glob("member_*"):
            member_id = int(member_path.name.split("_")[-1])
            all_models_results["id"].append(member_id)
            results_path = member_path / "results.json"
            with results_path.open("r") as f:
                member_results = json.load(f)

            for key, value in traverse(member_results, include_sequences=False):
                all_models_results[key].append(value)

        logger.info(
            f"Computing best models based on the selection criteria: init: {training_job.best_init}, fold: {training_job.best_fold}"
        )
        all_model_results_df = pl.DataFrame(all_models_results).with_columns(
            pl.col("fit.start").str.to_datetime(), pl.col("fit.end").str.to_datetime()
        )
        all_model_results_df.write_parquet(self.all_models_results_path)

        bins_cols = [
            "et_bin.low",
            "et_bin.high",
            "et_bin.closed",
            "eta_bin.low",
            "eta_bin.high",
            "eta_bin.closed",
        ]
        best_init_results = (
            all_model_results_df
            .group_by((*bins_cols, "fold"))
            .agg(pl.all().pipe(training_job.best_init.filter_polars_expr))
        )
        selected_models = (
            best_init_results
            .group_by(*bins_cols)
            .agg(pl.all().pipe(training_job.best_fold.filter_polars_expr))
        )
        selected_models.write_parquet(self.selected_models_path)
    
    @classmethod
    def load(cls, path: Path | str) -> Self:
        cls.validate_saved_directory(path)
        path = Path(path)
        config_path = path / "config.json"
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        instance = cls(**config)
        return instance

    @staticmethod
    def validate_saved_directory(output_path: Path | str):
        output_path = Path(output_path)
        if not output_path.exists():
            raise FileNotFoundError(f"Path {output_path} does not exist.")
        if not output_path.is_dir():
            raise NotADirectoryError(f"Path {output_path} is not a directory.")
        if not (output_path / "config.json").exists():
            raise FileNotFoundError(f"Config file not found in {output_path}.")
        if not (output_path / "all_models_results.parquet").exists():
            raise FileNotFoundError(
                f"All models results file not found in {output_path}."
            )
        if not (output_path / "selected_models.parquet").exists():
            raise FileNotFoundError(f"Selected models file not found in {output_path}.")

        for member_output_path in output_path.glob("member_*"):
            PTQConversionJob.validate_saved_member_directory(member_output_path)

    @staticmethod
    def validate_saved_member_directory(member_output_path: Path | str):
        member_output_path = Path(member_output_path)
        if not member_output_path.exists():
            raise FileNotFoundError(
                f"Member directory {member_output_path} does not exist."
            )
        if not member_output_path.is_dir():
            raise NotADirectoryError(
                f"Member directory {member_output_path} is not a directory."
            )

        model_path = member_output_path / "model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found in {member_output_path}.")
        results_path = member_output_path / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found in {member_output_path}.")

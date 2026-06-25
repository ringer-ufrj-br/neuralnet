from pydantic import BaseModel, Field, ConfigDict, validate_call
from pathlib import Path
import logging
from typing import Annotated, Literal
import polars as pl
from .jobs import MLPKerasTrainingJob
from .dataset import (
    DataTableType,
    RingsColType,
    EtColType,
    EtaColType,
)
from ...datasets import DirectoryType, ParquetDataset
from ...pydantic import YamlBaseModel


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


type PlaceType = Literal["kernel", "bias"]


class HGQFixedPointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    i0: Annotated[
        int,
        Field(
            description="Number of integer bits for the fixed-point representation.",
        ),
    ]
    f0: Annotated[
        int,
        Field(
            description="Number of fractional bits for the fixed-point representation.",
        ),
    ]

    @validate_call
    def as_hgq_quantizer_config(self, place: PlaceType):
        from hgq.config import QuantizerConfig
        from hgq.constraints import Constant

        return QuantizerConfig(
            q_type="kif",
            place=place,
            i0=self.i0,
            f0=self.f0,
            fc=Constant(self.f0),
            ic=Constant(self.i0),
        )


class UniformPTQInferenceJob(InferenceJob):
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

    def submit(self) -> pl.LazyFrame:
        logger = logging.getLogger(self.logger_name)
        logger.info(
            f"Starting uniform PTQ inference job with fit_job_path {self.fit_job_path}"
        )
        training_job = MLPKerasTrainingJob.load(self.fit_job_path)
        parquet_dataset = ParquetDataset(dataset_dir=self.dataset_dir)
        data_table = parquet_dataset.get_dataframe(self.data_table)
        committee_model = training_job.get_committee_model(
            et_col=self.et_col, eta_col=self.eta_col, rings_col=self.rings_col
        )
        committee_model.quantize(
            weight_quantizer_config=self.weight_quantization.as_hgq_quantizer_config(
                place="kernel"
            ),
            bias_quantizer_config=self.bias_quantization.as_hgq_quantizer_config(
                place="bias"
            ),
        )
        logger.info("Predicting")
        prediction_df = committee_model.predict(data_table)
        output_table_path = parquet_dataset.get_table_path(self.output_table)
        logger.info(f"Writing predictions to {output_table_path}")
        prediction_df.write_parquet(
            pl.PartitionBy(
                str(output_table_path),
                max_rows_per_file=self.max_rows_per_file,
            )
        )
        return prediction_df

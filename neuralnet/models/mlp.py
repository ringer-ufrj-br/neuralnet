from collections import defaultdict
from itertools import product
from pathlib import Path
import pickle
from typing import Annotated, Literal
import numpy as np
from keras import Sequential, Model
from keras.models import load_model
from keras.layers import Dense
from pydantic import BaseModel, Field, ConfigDict, computed_field
import typer
import yaml
import polars as pl
from functools import cached_property

from ..tensorflow.tunning import training, RefType
from .. import get_logger
from ..datasets import ParquetDataset
from ..datasets.ringer import get_ring_slices_per_layer
from ..submitit import ExecutorConfig
from ..utils import pydantic_to_markdown_schema


class DenseLayer(BaseModel):
    units: int
    activation: str
    kernel_initializer: str = Field(default="glorot_uniform")
    object_type: Literal["dense"] = Field(default="dense", description='Layer name. Must be "dense"')
    bias_initializer: str = Field(default="zeros")

    def get(self) -> Dense:
        return Dense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )


def get_model() -> Sequential:
    from keras import Input

    model = Sequential(
        [
            Input(shape=(50,)),
            Dense(5, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model


def get_n_folds(kfold_table_glob: str, fold_col: str) -> int:
    n_folds = (
        pl.scan_parquet(kfold_table_glob)
        .filter(pl.col(fold_col).is_not_null())
        .select(pl.col(fold_col).max().alias("max_fold"))
        .collect()
        .item()
    )
    return n_folds + 1  # Folds are 0-indexed


def norm1(data):
    norms = np.abs(data.sum(axis=1))
    norms[norms == 0] = 1
    return data / norms[:, None]


class VariableBin(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    col: pl.Expr | str
    lower: float
    upper: float
    closed: Literal["left", "right", "both", "none"] = "left"

    def model_post_init(self, context):
        if isinstance(self.col, str):
            self.col = pl.col(self.col)
        return super().model_post_init(context)

    @computed_field(
        repr=False,
        description="Polars condition for this bin",
    )
    @cached_property
    def is_inside_bin_polars(self) -> pl.Expr:
        return self.col.is_between(self.lower, self.upper, closed=self.closed)

    def is_inside_numpy(self, value):
        if self.closed == "left":
            return self.lower <= value < self.upper
        elif self.closed == "right":
            return self.lower < value <= self.upper
        elif self.closed == "both":
            return self.lower <= value <= self.upper
        else:
            return self.lower < value < self.upper


class BinnedKerasModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bins: list[VariableBin]
    features: list[str | pl.Expr]
    model: Path | Model

    def model_post_init(self, context):
        if isinstance(self.model, Path):
            self.model = load_model(self.model)
        self.features = [
            pl.col(feature) if isinstance(feature, str) else feature
            for feature in self.features
        ]

    @computed_field(
        repr=False,
        description="Polars condition for this model",
    )
    @cached_property
    def valid_bin_polars_expr(self) -> pl.Expr:
        return pl.all_horizontal([bin.is_inside_bin_polars for bin in self.bins])

    @computed_field(
        repr=False,
        description="Polars expression for the model prediction",
    )
    @cached_property
    def predict_polars_expr(self) -> pl.Expr:
        return self.input_col.map_batches(
            self.predict_polars_batch, return_dtype=pl.Float32
        )

    def predict_polars_batch(self, batch: pl.Series) -> pl.Series:
        data = norm1(np.stack(batch.to_numpy())).astype(np.float32)
        prediction = self.model.predict(data)
        return pl.Series(prediction.flatten(), dtype=pl.Float32)

    def predict(self, data: np.ndarray) -> np.ndarray:
        data = norm1(data).astype(np.float32)
        prediction = self.model.predict(data)
        return prediction.flatten()


class BinnedKerasMoE(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    models: list[BinnedKerasModel]

    @computed_field(
        repr=False,
        description="Polars expression for the MoE prediction",
    )
    @cached_property
    def predict_polars_expr(self) -> pl.Expr:
        first_model = self.models[0]
        prediction_col = pl.when(
            first_model.valid_bin_polars_expr,
        ).then(first_model.predict_polars_expr)
        for model in self.models[1:]:
            prediction_col = prediction_col.when(model.valid_bin_polars_expr).then(
                model.predict_polars_expr
            )
        prediction_col = prediction_col.otherwise(pl.lit(None, dtype=pl.Float32))
        return prediction_col

    def predict(self, data: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
        prediction_df = []
        for model in self.models:
            filter_condition = pl.all_horizontal(
                *(feature.is_not_null() for feature in model.features),
                model.valid_bin_polars_expr,
            )
            filtered = data.filter(filter_condition).select("id", *model.features)
            if isinstance(filtered, pl.LazyFrame):
                filtered = filtered.collect()
            if filtered.is_empty():
                filtered.clear()  # Frees memory premptively
                continue
            features = filtered.select(pl.exclude("id")).to_numpy()
            filtered = filtered.drop(pl.exclude("id"))
            prediction = model.predict(features).astype(np.float32)
            del features  # Frees memory premptively
            filtered = filtered.with_columns(pl.Series(prediction).alias("prediction"))
            prediction_df.append(filtered)
        return pl.concat(prediction_df)


class MLPTrainingJob(BaseModel):
    """
    Job for training a MLP model on a given dataset, with a given configuration.
    """

    dataset_dir: Annotated[
        Path, Field(description="Directory containing the parquet dataset")
    ]
    data_table: Annotated[
        str, Field(description="Name of the data table in the parquet dataset")
    ]
    rings_col: Annotated[
        str, Field(description="Name of the rings column in the data table")
    ]
    kfold_table: Annotated[
        str, Field(description="Name of the kfold table in the parquet dataset")
    ]
    label_col: Annotated[
        str, Field(description="Name of the label column in the kfold table")
    ]
    fold_col: Annotated[
        str, Field(description="Name of the fold column in the kfold table")
    ]
    et_col: Annotated[str, Field(description="Name of the et column in the data table")]
    et_bins: Annotated[
        str | tuple[float, ...],
        Field(
            description='et bin to select from the dataset. Should be in the format "et_bin_left,et_bin_right". Example: "20000,40000"'
        ),
    ]
    eta_col: Annotated[
        str, Field(description="Name of the eta column in the data table")
    ]
    eta_bins: Annotated[
        str | tuple[float, ...],
        Field(
            description='eta bin to select from the dataset. Should be in the format "eta_bin_left,eta_bin_right". Example: "0.0,0.8"'
        ),
    ]
    output_dir: Annotated[
        Path | None, Field(description="Directory to save the training results")
    ] = None
    tag: Annotated[str, Field(description="Tag to identify the training run")]
    batch_size: Annotated[int, Field(description="Batch size for training")] = 1024
    inits: Annotated[int, Field(description="Number of initializations")] = 5
    dry_run: Annotated[
        bool, Field(description="Perform a dry run without actually training")
    ] = False
    executor_config: Annotated[
        ExecutorConfig,
        Field(
            description="Slurm configuration for running the training job on a Slurm cluster"
        ),
    ]

    def model_post_init(self, context):
        if isinstance(self.et_bins, str):
            self.et_bins = tuple(float(x) for x in self.et_bins.split(","))
        if isinstance(self.eta_bins, str):
            self.eta_bins = tuple(float(x) for x in self.eta_bins.split(","))

        if self.output_dir is None:
            self.output_dir = Path.cwd() / "mlp_training_job"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return super().model_post_init(context)

    @classmethod
    def from_yaml(cls, yaml_file: Path, **kwargs) -> "MLPTrainingJob":
        """Load MLPTrainingJob from a YAML file."""
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        for key, value in kwargs.items():
            data[key] = value
        return cls(**data)

    def get_data(
        self,
        ring_indexes: list[int],
        data_table_glob: Path,
        kfold_table_glob: Path,
        fold: int,
        et_bin_left: float,
        et_bin_right: float,
        eta_bin_left: float,
        eta_bin_right: float,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:

        et = pl.col(self.et_col)
        et_bin_left = pl.lit(et_bin_left, dtype=pl.dtype_of(et))
        et_bin_right = pl.lit(et_bin_right, dtype=pl.dtype_of(et))

        eta = pl.col(self.eta_col).abs()
        eta_bin_left = pl.lit(eta_bin_left, dtype=pl.dtype_of(eta))
        eta_bin_right = pl.lit(eta_bin_right, dtype=pl.dtype_of(eta))

        # rings_norm = pl.col(self.rings_col).list.gather(
        #     ring_indexes).list.sum().abs()
        # rings_norm = pl.when(rings_norm == 0).then(1).otherwise(rings_norm)
        rings = [
            pl.col(self.rings_col).list.get(i).alias(f"rings_{i}") for i in ring_indexes
        ]

        data_df = (
            pl.scan_parquet(data_table_glob)
            .filter(
                et.is_between(et_bin_left, et_bin_right, closed="left")
                & eta.is_between(eta_bin_left, eta_bin_right, closed="left")
            )
            .select("id", *rings)
        )

        label = pl.col(self.label_col)

        fold_col = pl.col(self.fold_col)
        fold = pl.lit(fold, dtype=pl.dtype_of(fold_col))
        val_fold_df = (
            pl.scan_parquet(kfold_table_glob)
            .filter((fold_col == fold) & label.is_not_null())
            .select("id", label.cast(pl.Int32))
        )

        train_fold_df = (
            pl.scan_parquet(kfold_table_glob)
            .filter((fold_col != fold) & label.is_not_null())
            .select("id", label.cast(pl.Int32))
        )

        train_df = data_df.join(train_fold_df, on="id", how="inner").drop("id")
        val_df = data_df.join(val_fold_df, on="id", how="inner").drop("id")

        return train_df, val_df

    def load_data(
        self,
        fold: int,
        et_bin_left: float,
        et_bin_right: float,
        eta_bin_left: float,
        eta_bin_right: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load training and validation data for the given fold."""

        ring_indexes = get_ring_slices_per_layer(fraction=2)
        dataset = ParquetDataset(dataset_dir=self.dataset_dir)

        train_df, val_df = self.get_data(
            ring_indexes=ring_indexes,
            data_table_glob=dataset.get_table_glob(self.data_table),
            kfold_table_glob=dataset.get_table_glob(self.kfold_table),
            fold=fold,
            et_bin_left=et_bin_left,
            et_bin_right=et_bin_right,
            eta_bin_left=eta_bin_left,
            eta_bin_right=eta_bin_right,
        )
        train_df, val_df = pl.collect_all([train_df, val_df])

        # The dataframes only have rings and the label,
        # we need to separate them and convert the rings to normalized and quantized numpy arrays
        val_label = val_df.drop_in_place(self.label_col).to_numpy().flatten()
        val_rings = val_df.to_numpy()
        del val_df  # Frees memory premptively
        val_rings = norm1(val_rings)
        train_label = train_df.drop_in_place(self.label_col).to_numpy().flatten()
        train_rings = train_df.to_numpy()
        del train_df  # Frees memory premptively
        train_rings = norm1(train_rings)

        return train_rings, val_rings, train_label, val_label

    def load_ref(
        self,
        et_bin_left: float,
        et_bin_right: float,
        eta_bin_left: float,
        eta_bin_right: float,
    ) -> RefType:
        dataset = ParquetDataset(dataset_dir=self.dataset_dir)
        ref_df = (
            pl.scan_parquet(dataset.get_table_glob("ref"))
            .filter(
                (pl.col("et_bin_lower") == et_bin_left)
                & (pl.col("et_bin_upper") == et_bin_right)
                & (pl.col("eta_bin_lower") == eta_bin_left)
                & (pl.col("eta_bin_upper") == eta_bin_right)
            )
            .collect()
        )
        ref = defaultdict(lambda: defaultdict(dict))
        for row in ref_df.iter_rows(named=True):
            ref[row["criteria"]][row["sample_type"]][row["total_or_passed"]] = row[
                "value"
            ]
        for key in ref:
            ref[key] = dict(ref[key])
        ref = dict(ref)

        return ref

    def run_training(
        self,
        et_bin_left: float,
        et_bin_right: float,
        eta_bin_left: float,
        eta_bin_right: float,
        fold: int,
        init: int,
    ):

        logger = get_logger()
        logger.info(
            f"Loading data for et_bin ({et_bin_left}, {et_bin_right}), eta_bin ({eta_bin_left}, {eta_bin_right}), fold {fold} and init {init}"
        )
        ref = self.load_ref(
            et_bin_left=et_bin_left,
            et_bin_right=et_bin_right,
            eta_bin_left=eta_bin_left,
            eta_bin_right=eta_bin_right,
        )
        X, X_val, y, y_val = self.load_data(
            fold=fold,
            et_bin_left=et_bin_left,
            et_bin_right=et_bin_right,
            eta_bin_left=eta_bin_left,
            eta_bin_right=eta_bin_right,
        )
        output_dir = (
            self.output_dir
            / f"et_{et_bin_left}_{et_bin_right}"
            / f"eta_{eta_bin_left}_{eta_bin_right}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        training(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            sort=fold,
            init=init,
            tag=self.tag,
            loss="binary_crossentropy",
            verbose=True,
            ref=ref,
            model=get_model(),
            output_dir=str(output_dir),
            batch_size=self.batch_size,
            dry_run=self.dry_run,
            et_bin=(et_bin_left, et_bin_right),
            eta_bin=(eta_bin_left, eta_bin_right),
        )
        logger.info(
            f"Training completed for et_bin ({et_bin_left}, {et_bin_right}), eta_bin ({eta_bin_left}, {eta_bin_right}), fold {fold} and init {init}"
        )

    def run(self):
        logger = get_logger()
        dataset = ParquetDataset(dataset_dir=str(self.dataset_dir))
        n_folds = get_n_folds(
            kfold_table_glob=dataset.get_table_glob(self.kfold_table),
            fold_col=self.fold_col,
        )
        logger.info(f"The dataset has {n_folds} folds.")
        folds_range = range(n_folds)
        inits_range = range(self.inits)
        et_bins_iterator = zip(
            self.et_bins[:-1],
            self.et_bins[1:],
        )
        eta_bins_iterator = zip(
            self.eta_bins[:-1],
            self.eta_bins[1:],
        )
        executor = self.executor_config.get_executor()
        bins_iterator = product(et_bins_iterator, eta_bins_iterator)
        i = 0
        with executor.batch():
            for (et_bin_left, et_bin_right), (
                eta_bin_left,
                eta_bin_right,
            ) in bins_iterator:
                if i > 0 and self.dry_run:
                    logger.info("Dry run enabled, stopping after first bin.")
                    break
                for fold, init in product(folds_range, inits_range):
                    logger.info(
                        f"{i} - Submitting training job for et_bin ({et_bin_left}, {et_bin_right}), eta_bin ({eta_bin_left}, {eta_bin_right}), fold {fold} and init {init}"
                    )
                    executor.submit(
                        self.run_training,
                        et_bin_left,
                        et_bin_right,
                        eta_bin_left,
                        eta_bin_right,
                        fold,
                        init,
                    )
                    i += 1

        logger.info("All training jobs submitted.")

    @staticmethod
    def load_model(
        results_dir: Path, eta_col: str, et_col: str, rings_col: str
    ) -> tuple[pl.DataFrame, pl.DataFrame, BinnedKerasMoE]:
        results = []
        logger = get_logger()
        expected_cols = {
            "et_bin_lower",
            "et_bin_upper",
            "eta_bin_lower",
            "eta_bin_upper",
            "sort",
            "init",
            "tag",
            "model",
            "time",
        }

        for i, individual_fit_dir in enumerate(results_dir.glob("*/*/*")):
            logger.info(f"{i} - Processing {individual_fit_dir}")
            with open(individual_fit_dir / "results.pic", "rb") as f:
                d = pickle.load(f)
            record = {}
            record["et_bin_lower"] = float(d["metadata"].get("et_bin")[0])
            record["et_bin_upper"] = float(d["metadata"].get("et_bin")[1])
            record["eta_bin_lower"] = float(d["metadata"].get("eta_bin")[0])
            record["eta_bin_upper"] = float(d["metadata"].get("eta_bin")[1])
            record["sort"] = int(d["metadata"].get("sort"))
            record["init"] = int(d["metadata"].get("init"))
            record["tag"] = d["metadata"].get("tag")
            record["model"] = str(individual_fit_dir / "model.keras")
            record["time"] = d.get("time")

            summary_dict = d["history"].pop("summary")
            if not summary_dict:
                raise ValueError(
                    f"Summary dictionary is empty for {individual_fit_dir}"
                )
            for key, value in summary_dict.items():
                if isinstance(value, (int, float, str)):
                    record[f"summary.{key}"] = value
                    expected_cols.add(f"summary.{key}")
                elif isinstance(value, tuple):
                    metric, approved, total = value
                    record[f"summary.{key}"] = metric
                    record[f"summary.{key}.approved"] = approved
                    record[f"summary.{key}.total"] = total
                    expected_cols.update(
                        {
                            f"summary.{key}",
                            f"summary.{key}.approved",
                            f"summary.{key}.total",
                        }
                    )
                else:
                    raise ValueError(
                        f"Unsupported type for summary key {key}: {type(value)}"
                    )

            reference_dict = d["history"].pop("reference")
            if reference_dict:
                for criteria, criteria_metrics in reference_dict.items():
                    for metric, metric_values in criteria_metrics.items():
                        if isinstance(metric_values, (int, float, str)):
                            col = f"reference.{criteria}.{metric}"
                            record[col] = metric_values
                            expected_cols.add(col)
                        elif isinstance(metric_values, tuple):
                            value, approved, total = metric_values
                            value_col = f"reference.{criteria}.{metric}"
                            approved_col = f"reference.{criteria}.{metric}.approved"
                            total_col = f"reference.{criteria}.{metric}.total"
                            record[value_col] = value
                            record[approved_col] = approved
                            record[total_col] = total
                            expected_cols.update({value_col, approved_col, total_col})
            else:
                logger.warning(
                    f"Reference dictionary is empty for {individual_fit_dir}"
                )

            for metric_name, metric_values in d["history"].items():
                if isinstance(metric_values, list) and all(
                    isinstance(v, (int, float)) for v in metric_values
                ):
                    record[f"history.{metric_name}"] = metric_values
                elif isinstance(metric_values, (int, float)):
                    record[f"history.{metric_name}"] = metric_values
                else:
                    logger.warning(
                        f"Skipping metric {metric_name} with non-numeric values for {individual_fit_dir}"
                    )
            results.append(record)

        for record in results:
            for col in expected_cols:
                if col not in record:
                    record[col] = None

        results = (
            pl.from_records(results, infer_schema_length=1000)
            .sort(["et_bin_lower", "eta_bin_lower", "sort", "init"])
            .with_row_index("id")
        )
        best_models = (
            results.group_by(
                ["et_bin_lower", "et_bin_upper", "eta_bin_lower", "eta_bin_upper"]
            )
            .agg(pl.all().sort_by("summary.max_sp_val", descending=True).first())
            .sort("id")
        )

        ring_indexes = get_ring_slices_per_layer(fraction=2)
        models = []
        for row in best_models.iter_rows(named=True):
            bins = [
                dict(
                    col=pl.col(eta_col).abs(),
                    lower=row["eta_bin_lower"],
                    upper=row["eta_bin_upper"],
                    closed="left",
                ),
                dict(
                    col=pl.col(et_col),
                    lower=row["et_bin_lower"],
                    upper=row["et_bin_upper"],
                    closed="left",
                ),
            ]
            model = dict(
                bins=bins,
                features=[
                    pl.col(rings_col)
                    .list.get(ring_index)
                    .alias(f"{rings_col}[{ring_index}]")
                    for ring_index in ring_indexes
                ],
                model=row["model"],
            )
            models.append(model)
        selected_model = BinnedKerasMoE(models=models)
        return results, best_models, selected_model


app = typer.Typer(help="Ringer Zero MLP commands", rich_markup_mode="markdown")


RUN_TRAINING_HELP = "Run MLP training jobs"


@app.command(
    short_help=RUN_TRAINING_HELP,
    help=f"**{RUN_TRAINING_HELP}**\n\n{pydantic_to_markdown_schema(MLPTrainingJob)}",
)
def run_training(
    config: Annotated[
        Path,
        typer.Option(
            "--config", help="Path to the YAML configuration file for the training job"
        ),
    ],
):
    job = MLPTrainingJob.from_yaml(config)
    job.run()


@app.command()
def add_inference(
    dataset_dir: Annotated[
        Path,
        typer.Option("--dataset-dir", help="Directory containing the parquet dataset"),
    ],
    results_dir: Annotated[
        Path,
        typer.Option(
            "--results-dir",
            help="Directory containing the training results to add inference to",
        ),
    ],
    features_table: Annotated[
        str,
        typer.Option(
            "--features-table",
            help="Name of the table containing the features to run inference on",
        ),
    ],
    inference_table: Annotated[
        str,
        typer.Option(
            "--inference-table", help="Name of the table to save the inference results"
        ),
    ],
    eta_col: Annotated[
        str, typer.Option("--eta-col", help="Name of the eta column in the data table")
    ] = "TrigEMClusterContainer.eta",
    et_col: Annotated[
        str, typer.Option("--et-col", help="Name of the et column in the data table")
    ] = "TrigEMClusterContainer.et",
    rings_col: Annotated[
        str,
        typer.Option("--rings-col", help="Name of the rings column in the data table"),
    ] = "TrigEMClusterContainer.ringsE",
):
    results, best_models, loaded_model = MLPTrainingJob.load_model(
        results_dir, eta_col, et_col, rings_col
    )
    results.write_parquet(results_dir / "all_results.parquet")
    best_models.write_parquet(results_dir / "best_models.parquet")
    results.clear()  # Frees memory premptively
    best_models.clear()  # Frees memory premptively
    parquet_dataset = ParquetDataset(dataset_dir=dataset_dir)
    features_df = pl.scan_parquet(parquet_dataset.get_table_glob(features_table))
    prediction_df = loaded_model.predict(features_df)
    prediction_df.write_parquet(
        pl.PartitionBy(
            str(parquet_dataset.get_table_path(inference_table)),
            max_rows_per_file=100_000,
        ),
        compression="snappy",
    )
    return prediction_df

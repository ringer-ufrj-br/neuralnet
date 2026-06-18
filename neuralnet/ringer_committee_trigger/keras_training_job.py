from collections import defaultdict

import polars as pl
from typing import Annotated, Any, Literal, NotRequired, Self, TypedDict
from itertools import product
from pydantic import (
    Field,
    ConfigDict,
    PrivateAttr,
    BaseModel,
    computed_field,
    validate_call,
)
import logging
from pathlib import Path
import numpy as np
import numpy.typing as npt
from functools import cached_property
import json
from zipfile import ZipFile, ZIP_DEFLATED
from ..submitit import ExecutorConfig
from ..logging import LoggerName
from .dataset import (
    RingerParquetDataset,
    BatchSizeType,
    DataTableType,
    RingsColType,
    KFoldTableType,
    LabelColType,
    FoldColType,
    RingFractionType,
    NormStrategyType,
    EtColType,
    EtaColType,
    Bin,
)
from ..numpy import inverse_sigmoid
from ..metrics import enhanced_confusion_matrix, EnhancedConfusionMatrixDict
from ..models.keras.factories import (
    EpochsType,
    VerboseType,
    FitRoutineDict,
    KerasSequentialModelFactory,
    LossType,
    OptimizerType,
)
from ..utils import traverse
from ..polars import PolarsExpression
from ..datasets import DirectoryType
from ..json import cast_to_json_value

type RingerTrainingJobDatasetType = Annotated[
    RingerParquetDataset,
    Field(
        description="Dataset to be used.",
        discriminator="object_type",
    ),
]

type FromLogitsType = Annotated[
    bool,
    Field(
        False,
        description="Whether the model output to consider is logits. When enabled considers the threshold limits as probabilities between 0 and 1",
    ),
]


class FitMetricsDict(TypedDict):
    loss: list[float]
    accuracy: list[float]
    val_max_sp: list[float]
    val_max_sp_pd: list[float]
    val_max_sp_fa: list[float]
    val_max_sp_threshold: list[float]


class FitDict(FitRoutineDict):
    train: FitMetricsDict
    val: FitMetricsDict


class EvaluationDict(EnhancedConfusionMatrixDict):
    loss: float
    weighted: NotRequired[EnhancedConfusionMatrixDict]


class RingerCommitteeKerasTrainingJobResults(TypedDict):
    fold: int
    init: int
    et_bin: Bin
    eta_bin: Bin
    fit: FitDict
    train: EvaluationDict
    val: EvaluationDict
    test: EvaluationDict


type PatienceType = Annotated[
    int,
    Field(
        25,
        description="Number of epochs with no improvement after which training will be stopped.",
    ),
]


class SelectionCriteria(BaseModel):
    key: str = Field(
        "val.max_sp.sp",
        description="Criterion to select the best model initilization or fold groups. Must be a key in the fit dictionary of the training results.",
    )
    criterion: Literal["max", "min"] = Field(
        "max",
        description="How to select the best model from initilization or fold groups. Must be either 'max' or 'min'.",
    )


type EtaBinIntervalValue = Annotated[
    float,
    Field(
        ge=0,
        description="Represents the value of the Eta bin interval. Must be a non-negative integer.",
    ),
]


class RingerCommitteeKerasTrainingJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Dataset params
    batch_size: BatchSizeType
    data_table: DataTableType
    dataset_dir: DirectoryType
    et_col: EtColType
    et_bins: list[float] = Field(
        ...,
        description="Bins to be used for the Et variable. Must be a list of increasing values.",
        min_length=2,
    )
    eta_col: EtaColType
    eta_bins: list[EtaBinIntervalValue] = Field(
        ...,
        description="Bins to be used for the Eta variable. Must be a list of increasing values.",
        min_length=2,
    )
    fold_col: FoldColType
    kfold_table: KFoldTableType
    label_col: LabelColType
    norm_strategy: NormStrategyType
    rings_col: RingsColType
    ring_fraction: RingFractionType

    # Model params
    model_factory: KerasSequentialModelFactory = Field()
    from_logits: FromLogitsType
    epochs: EpochsType
    verbose: VerboseType
    loss: LossType
    optimizer: OptimizerType
    balance_class_weights: bool = Field(
        True,
        description="Whether to balance class weights during training. If True, the class weights will be set inversely proportional to the class frequencies.",
    )
    inits: Annotated[int, Field(description="Number of initializations")] = 5
    patience: PatienceType

    # Threshold Fit
    num_thresholds: int = Field(
        ge=2,
        description="Number of thresholds to evaluate for the strategy.",
    )
    lower_threshold: float = Field(
        ge=0,
        lt=1,
        description="Lower threshold limit for the strategy.",
    )
    upper_threshold: float = Field(
        gt=0,
        le=1,
        description="Upper threshold limit for the strategy.",
    )

    # Execution related fields
    dry_run: Annotated[
        bool, Field(description="Perform a dry run without actually training")
    ] = False
    executor_config: Annotated[
        ExecutorConfig,
        Field(
            description="Slurm configuration for running the training job on a Slurm cluster"
        ),
    ]
    output_path: Annotated[
        Path, Field(description="Path to save the results of the job")
    ]

    # Misc
    logger_name: LoggerName

    # Model Selection
    best_init: SelectionCriteria = Field(
        default_factory=SelectionCriteria,
        description="Criterion to select the best model initialization. Must be a key in the fit dictionary of the training results.",
    )
    best_fold: SelectionCriteria = Field(
        default_factory=SelectionCriteria,
        description="Criterion to select the best fold. Must be a key in the fit dictionary of the training results.",
    )

    # Private
    _thresholds: list[float] | None = PrivateAttr(default=None)

    @property
    def thresholds(self) -> list[float]:
        if self._thresholds is None:
            raise ValueError("Thresholds not initialized")
        return self._thresholds

    @cached_property
    def thresholds_array(self) -> npt.NDArray[np.floating]:
        return np.array(self.thresholds)

    @cached_property
    def config_path(self) -> Path:
        return self.output_path / "config.json"

    @cached_property
    def all_models_results_path(self) -> Path:
        return self.output_path / "all_models_results.parquet"

    @cached_property
    def selected_models_path(self) -> Path:
        return self.output_path / "selected_models.parquet"

    def model_post_init(self, context):
        res = super().model_post_init(context)
        if self.lower_threshold >= self.upper_threshold:
            raise ValueError(
                f"lower_threshold must be less than upper_threshold. Got lower_threshold={self.lower_threshold} and upper_threshold={self.upper_threshold}."
            )
        self._thresholds = np.linspace(
            self.lower_threshold,
            self.upper_threshold,
            self.num_thresholds,
        )
        if self.from_logits:
            self._thresholds = inverse_sigmoid(self._thresholds)

        self._thresholds = self._thresholds.tolist()  # Convert to list for JSON serialization
        self.et_bins.sort()
        self.eta_bins.sort()

        return res

    def submit(self):

        logger = logging.getLogger(self.logger_name)
        logger.info(
            f"Starting job with model {self.model_factory.name} on dataset at {self.dataset_dir}"
        )
        self.output_path.mkdir(parents=True)
        self.output_path.joinpath("config.json").write_text(
            self.model_dump_json(indent=4, exclude={"models", "results"}),
            encoding="utf-8",
        )

        dataset = RingerParquetDataset(
            dataset_dir=self.dataset_dir,
            data_table=self.data_table,
            rings_col=self.rings_col,
            kfold_table=self.kfold_table,
            label_col=self.label_col,
            fold_col=self.fold_col,
            et_col=self.et_col,
            eta_col=self.eta_col,
            ring_fraction=self.ring_fraction,
            norm_strategy=self.norm_strategy,
        )
        n_folds = dataset.get_n_folds()
        logger.info(f"The dataset has {n_folds} folds.")

        et_bin_idxs = range(len(self.et_bins) - 1)
        eta_bin_idxs = range(len(self.eta_bins) - 1)
        folds_range = range(n_folds)
        inits_range = range(self.inits)
        iterator = product(et_bin_idxs, eta_bin_idxs, folds_range, inits_range)
        executor = self.executor_config.get_executor()
        members = {
            "id": [],
            "et_bin.low": [],
            "et_bin.high": [],
            "et_bin.closed": [],
            "eta_bin.low": [],
            "eta_bin.high": [],
            "eta_bin.closed": [],
            "fold": [],
            "init": [],
        }
        submitted_jobs = []
        with executor.batch():
            for member_id, (et_bin_idx, eta_bin_idx, fold, init) in enumerate(iterator):
                et_bin = Bin(
                    low=self.et_bins[et_bin_idx],
                    high=self.et_bins[et_bin_idx + 1],
                    closed="left",
                )
                eta_bin = Bin(
                    low=self.eta_bins[eta_bin_idx],
                    high=self.eta_bins[eta_bin_idx + 1],
                    closed="left",
                )
                logger.info(
                    f"{member_id}: Submitting training for Et bin {et_bin} and Eta bin {eta_bin}, fold {fold} and init {init}"
                )
                submitted_job = executor.submit(
                    self._run_training,
                    member_id,
                    et_bin.model_dump(),
                    eta_bin.model_dump(),
                    fold,
                    init,
                )
                submitted_jobs.append(submitted_job)
                members["id"].append(member_id)
                members["fold"].append(fold)
                members["init"].append(init)
                members["et_bin.low"].append(et_bin.low)
                members["et_bin.high"].append(et_bin.high)
                members["et_bin.closed"].append(et_bin.closed)
                members["eta_bin.low"].append(eta_bin.low)
                members["eta_bin.high"].append(eta_bin.high)
                members["eta_bin.closed"].append(eta_bin.closed)

        dependent_executor = self.executor_config.get_executor()
        from submitit import AutoExecutor

        if isinstance(dependent_executor, AutoExecutor):
            dependency_string = ":".join(str(job.job_id) for job in submitted_jobs)
            logger.info(
                f"Submitting dependent job with dependency on jobs: {dependency_string}"
            )
            dependent_executor.update_parameters(
                slurm_additional_parameters={
                    "dependency": f"afterok:{dependency_string}"
                }
            )
        dependent_executor.submit(self.post_training, members)

        logger.info("All jobs submitted.")

    def _run_training(
        self, member_id: int, et_bin: dict, eta_bin: dict, fold: int, init: int
    ):
        from neuralnet.models.keras.routines import fit_routine, evaluation_routine
        from keras import Model

        logger = logging.getLogger(self.logger_name)
        logger.info(
            f"{member_id}: Running training for Et bin {et_bin} and Eta bin {eta_bin}, fold {fold} and init {init}"
        )
        dataset = RingerParquetDataset(
            dataset_dir=self.dataset_dir,
            data_table=self.data_table,
            rings_col=self.rings_col,
            kfold_table=self.kfold_table,
            label_col=self.label_col,
            fold_col=self.fold_col,
            fold=fold,
            et_col=self.et_col,
            et_bin=et_bin,
            eta_col=self.eta_col,
            eta_bin=eta_bin,
            ring_fraction=self.ring_fraction,
            norm_strategy=self.norm_strategy,
        )
        train_numpy = dataset.train_numpy()
        val_numpy = dataset.val_numpy()
        results: RingerCommitteeKerasTrainingJobResults = {
            "fold": fold,
            "init": init,
            "et_bin": et_bin,
            "eta_bin": eta_bin,
        }
        output_dir = self.get_member_output_dir(member_id)
        output_dir.mkdir(parents=True)
        model_factory = self.model_factory.model_copy(
            update=dict(name=f"{self.model_factory.name}_{member_id}")
        )
        model: Model = model_factory.as_keras()
        if self.balance_class_weights:
            class_weights = dataset.train_class_weights()
        else:
            class_weights = None

        if self.dry_run:
            logger.warning("Dry run enabled, running one epoch.")
            epochs = 1
        else:
            epochs = self.epochs

        from neuralnet.callbacks.keras import SP

        sp_callback = SP(
            validation_data=val_numpy,
            patience=self.patience,
            verbose=self.verbose,
            save_the_best=True,
        )
        callbacks = [sp_callback]
        train_data = dataset.train_numpy()
        val_data = dataset.val_numpy()

        model, results["fit"] = fit_routine(
            model=model,
            train_data=train_data,
            val_data=val_data,
            loss=self.loss.as_keras(),
            optimizer=self.optimizer.as_keras(),
            metrics=["accuracy"],
            callbacks=callbacks,
            epochs=epochs,
            verbose=self.verbose,
            batch_size=self.batch_size,
            # class_weight=class_weights,
        )
        logger.info(f"Finished training for fold {fold} and init {init}")

        model_path = output_dir / "model.keras"
        model.save(str(model_path))

        eval_dict = evaluation_routine(
            model=model,
            data=train_numpy,
            loss=self.loss.as_keras(),
            optimizer=self.optimizer.as_keras(),
            metrics=self.get_metrics(),
        )
        results["train"] = self.process_eval_dict(
            eval_dict, dataset=dataset, data_category="train"
        )
        del train_numpy

        eval_dict = evaluation_routine(
            model=model,
            data=val_numpy,
            loss=self.loss.as_keras(),
            optimizer=self.optimizer.as_keras(),
            metrics=self.get_metrics(),
        )
        results["val"] = self.process_eval_dict(
            eval_dict, dataset=dataset, data_category="val"
        )
        del val_numpy

        if not self.dry_run and hasattr(dataset, "test_numpy"):
            test_numpy = dataset.test_numpy()
            eval_dict = evaluation_routine(
                model=model,
                data=test_numpy,
                loss=self.loss.as_keras(),
                optimizer=self.optimizer.as_keras(),
                metrics=self.get_metrics(),
            )
            results["test"] = self.process_eval_dict(
                eval_dict, dataset=dataset, data_category="test"
            )
            del test_numpy

        logger.info(f"Finished evaluating for fold {fold} and init {init}")
        results_path = output_dir / "results.json"
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(cast_to_json_value(results), f, indent=4)

        zip_path = output_dir / "results.json.zip"
        with ZipFile(zip_path, "w", ZIP_DEFLATED) as zipf:
            zipf.write(results_path, arcname="results.json")

        logger.info(f"Saved results to {output_dir}")

        del model
        from keras.backend import clear_session

        clear_session()

    def get_metrics(self):
        from keras.metrics import (
            TrueNegatives,
            TruePositives,
            FalseNegatives,
            FalsePositives,
        )

        metrics = [
            TrueNegatives(
                name="true_negatives",
                thresholds=self.thresholds,
            ),
            TruePositives(
                name="true_positives",
                thresholds=self.thresholds,
            ),
            FalseNegatives(
                name="false_negatives",
                thresholds=self.thresholds,
            ),
            FalsePositives(
                name="false_positives",
                thresholds=self.thresholds,
            ),
        ]
        return metrics

    def process_eval_dict(
        self,
        eval_dict: dict,
        dataset: RingerParquetDataset,
        data_category: str,
    ) -> EvaluationDict:
        import numpy as np

        enhanced_cm_dict = enhanced_confusion_matrix(
            tn=np.array(eval_dict.pop("true_negatives")),
            tp=np.array(eval_dict.pop("true_positives")),
            fn=np.array(eval_dict.pop("false_negatives")),
            fp=np.array(eval_dict.pop("false_positives")),
            thresholds=self.thresholds_array,
        )
        if self.balance_class_weights:
            match data_category:
                case "train":
                    class_weights = dataset.train_class_weights()
                case "val":
                    class_weights = dataset.val_class_weights()
                case "test":
                    class_weights = dataset.test_class_weights()
                case _:
                    raise ValueError(f"Unknown data_category: {data_category}")
            weighted_eval_dict = {
                "tn": np.array(enhanced_cm_dict["tn"]) * class_weights[0],
                "tp": np.array(enhanced_cm_dict["tp"]) * class_weights[1],
                "fn": np.array(enhanced_cm_dict["fn"]) * class_weights[1],
                "fp": np.array(enhanced_cm_dict["fp"]) * class_weights[0],
            }
            weighted_enhanced_cm_dict = enhanced_confusion_matrix(
                tn=weighted_eval_dict["tn"],
                tp=weighted_eval_dict["tp"],
                fn=weighted_eval_dict["fn"],
                fp=weighted_eval_dict["fp"],
                thresholds=self.thresholds_array,
            )
            enhanced_cm_dict["weighted"] = weighted_enhanced_cm_dict
        return enhanced_cm_dict

    def post_training(self, members):
        members_df = pl.DataFrame(members)
        all_models_results = defaultdict(list)
        for row in members_df.iter_rows(named=True):
            member_id = row["id"]
            all_models_results["id"].append(member_id)
            results_path = self.get_member_output_dir(member_id) / "results.json.zip"
            with ZipFile(results_path, "r") as zipf:
                with zipf.open("results.json") as f:
                    member_results = json.load(f)

            for key, value in traverse(member_results, include_sequences=False):
                all_models_results[key].append(value)

        all_model_results_df = pl.DataFrame(all_models_results)
        all_model_results_df.write_parquet(self.all_models_results_path)

        bins_cols = [
            "et_bin.low",
            "et_bin.high",
            "et_bin.closed",
            "eta_bin.low",
            "eta_bin.high",
            "eta_bin.closed",
        ]
        best_init_results = all_model_results_df.filter(
            pl.col(self.best_init.key)
            == pl.col(self.best_init.key).max().over((*bins_cols, "fold"))
        )
        selected_models = best_init_results.filter(
            pl.col(self.best_fold.key)
            == pl.col(self.best_fold.key).max().over(bins_cols)
        )
        selected_models.write_parquet(self.selected_models_path)

    def get_member_output_dir(self, member_id: int) -> Path:
        return self.output_path / f"member_{member_id}"

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
            RingerCommitteeKerasTrainingJob.validate_saved_member_directory(
                member_output_path
            )

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

    @classmethod
    def load(cls, path: Path | str) -> Self:
        cls.validate_saved_directory(path)
        path = Path(path)
        config_path = path / "config.json"
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        instance = cls(**config)
        return instance

    @cached_property
    def all_model_results(self) -> pl.DataFrame:
        results_path = self.output_path / "all_models_results.parquet"
        if not results_path.exists():
            raise FileNotFoundError(
                f"All models results file not found at {results_path}"
            )
        return pl.read_parquet(results_path)

    @cached_property
    def selected_models(self) -> pl.DataFrame:
        selected_models_path = self.output_path / "selected_models.parquet"
        if not selected_models_path.exists():
            raise FileNotFoundError(
                f"Selected models file not found at {selected_models_path}"
            )
        return pl.read_parquet(selected_models_path)

    def get_member_model(self, member_id: int, with_results: bool = False):
        member_output_dir = self.get_member_output_dir(member_id)

        from keras.models import load_model
        from keras import Model

        model_path = member_output_dir / "model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model: Model = load_model(model_path)

        if not with_results:
            return model

        results_path = member_output_dir / "results.json.zip"
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found at {results_path}")
        with ZipFile(results_path, "r") as zipf:
            with zipf.open("results.json") as f:
                results = json.load(f)

        return model, results

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_committee_model(
        self,
        et_col: PolarsExpression | None = None,
        eta_col: PolarsExpression | None = None,
        rings_col: PolarsExpression | None = None,
    ):
        from keras.models import Model

        selected_models = []
        for row in self.selected_models.iter_rows(named=True):
            member_id = row["id"]
            keras_model: Model = self.get_member_model(member_id)
            if et_col is None:
                et_col = pl.col(self.et_col)

            if eta_col is None:
                eta_col = pl.col(self.eta_col)

            if rings_col is None:
                rings_col = pl.col(self.rings_col)

            model = BinnedKerasModel(
                bins=[
                    VariableBin(
                        col=et_col,
                        lower=row["et_bin.lower"],
                        upper=row["et_bin.upper"],
                        closed=row["et_bin.closed"],
                    ),
                    VariableBin(
                        col=eta_col,
                        lower=row["eta_bin.lower"],
                        upper=row["eta_bin.upper"],
                        closed=row["eta_bin.closed"],
                    ),
                ],
                keras_model=keras_model,
                preprocessing=self.norm_strategy,
                features=[self.rings_col],
            )
            selected_models.append(model)

        if not selected_models:
            raise ValueError("No selected models found.")

        return BinnedKerasModelSpecialistCommittee(models=selected_models)


class VariableBin(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    col: PolarsExpression
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
    keras_model: Any
    preprocessing: Literal["l1"]
    features: list[PolarsExpression]

    def model_post_init(self, context):
        self.features = [
            pl.col(feature) if isinstance(feature, str) else feature
            for feature in self.features
        ]

    @cached_property
    def row_selector(self) -> pl.Expr:
        return pl.all_horizontal(
            [bin.is_inside_bin_polars for bin in self.bins]
            + [feature.is_not_null() for feature in self.features]
        )

    @cached_property
    def batched_predict_polars(self) -> pl.Expr:
        return self.input_col.map_batches(
            self.predict_polars_batch, return_dtype=pl.Float32
        )

    @cached_property
    def predict_polars_expr(self) -> pl.Expr:
        """
        This is extremely slow, please don't use it unless you really need to.
        """
        return (
            pl.when(self.row_selector)
            .then(self.batched_predict_polars)
            .otherwise(pl.lit(None, dtype=pl.Float32))
        )

    def preprocessing(self, data: np.ndarray) -> np.ndarray:
        if self.preprocessing == "l1":
            from ..numpy import alternative_norm1

            return alternative_norm1(data)
        else:
            raise ValueError(f"Unknown preprocessing strategy: {self.preprocessing}")

    def predict_polars_batch(self, batch: pl.Series) -> pl.Series:
        data = np.stack(batch.to_numpy())
        prediction = self.predict_numpy(data)
        return pl.Series(prediction.flatten(), dtype=pl.Float32)

    def predict_numpy(self, data: np.ndarray) -> np.ndarray:
        data = self.preprocessing(data).astype(np.float32)
        prediction = self.model.predict(data)
        return prediction.flatten()


class BinnedKerasModelSpecialistCommittee(BaseModel):
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
            first_model.row_selector,
        ).then(first_model.batched_predict_polars)
        for model in self.models[1:]:
            prediction_col = prediction_col.when(model.row_selector).then(
                model.batched_predict_polars
            )
        prediction_col = prediction_col.otherwise(pl.lit(None, dtype=pl.Float32))
        return prediction_col

    def predict(self, data: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
        prediction_df = []
        for model in self.models:
            filtered = data.filter(model.row_selector).select("id", *model.features)
            if isinstance(filtered, pl.LazyFrame):
                filtered = filtered.collect()
            if filtered.is_empty():
                filtered.clear()  # Frees memory premptively
                del filtered
                continue
            features = filtered.select(pl.exclude("id")).to_numpy()
            filtered = filtered.drop(pl.exclude("id"))
            prediction = model.predict(features).astype(np.float32)
            del features  # Frees memory premptively
            filtered = filtered.with_columns(pl.Series(prediction).alias("prediction"))
            prediction_df.append(filtered)
        return pl.concat(prediction_df)

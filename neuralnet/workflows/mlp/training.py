from collections import defaultdict

import polars as pl
from typing import (
    Annotated,
    Literal,
    Self,
    TypedDict,
    TYPE_CHECKING,
)

from prompt_toolkit import history

if TYPE_CHECKING:
    from hgq.config import QuantizerConfig
    from keras import Sequential

from itertools import product
from pydantic import (
    AfterValidator,
    Field,
    ConfigDict,
    BaseModel,
)
import logging
from pathlib import Path
import numpy as np
from functools import cached_property
import json
from zipfile import ZipFile, ZIP_DEFLATED
from ...submitit import ExecutorConfig
from ...logging import LoggerName
from ...datasets.ringer import (
    RingerParquetDataset,
    DataTableType,
    RingsColType,
    KFoldTableType,
    LabelColType,
    FoldColType,
)
from ...metrics import (
    enhanced_confusion_matrix,
    EnhancedConfusionMatrixDict,
    enhanced_confusion_matrix_from_preds,
)
from ...models.keras.factories import (
    EpochsType,
    VerboseType,
    FitRoutineDict,
    LossType,
    OptimizerType,
)
from ...models.binned_committee import BinnedCommittee, BinnedModel, VariableBin
from ...models.dense import MLPFactory
from ...utils import traverse
from ...normalizers.polars import AlternativeNorm1
from ...utils.polars import RingSlicesPerLayer
from ...datasets import DirectoryType
from ...pydantic import YamlBaseModel
from ...numpy import sigmoid
from ...bins import (
    Bin,
    BinDict,
    AbsoluteBin,
    AbsoluteBinDict,
)

type BalanceClassWeightsType = Annotated[
    bool,
    Field(
        description="Whether to balance class weights during training. If True, the class weights will be set inversely proportional to the class frequencies.",
    ),
]

type BatchSizeType = Annotated[
    int,
    Field(
        gt=0,
        description="Batch size for the dataset. Must be a positive integer.",
    ),
]

type EtBinType = Annotated[
    list[Bin],
    Field(
        description="Bins to be used for the Et variable.",
        min_length=1,
    ),
]

type EtaBinType = Annotated[
    list[AbsoluteBin],
    Field(
        description="Bins to be used for the Eta variable.",
        min_length=1,
    ),
]

type EtColType = Annotated[
    str, Field(description="Name of the et column in the data table")
]

type EtaColType = Annotated[
    str, Field(description="Name of the eta column in the data table")
]

type FromLogitsType = Annotated[
    bool,
    Field(
        description="Whether the model output to consider is logits. When enabled considers the threshold limits as probabilities between 0 and 1",
    ),
]

type NormStrategyType = Annotated[
    Literal["l1"] | None,
    Field(
        description="Normalization strategy to apply to the rings. If None, no normalization is applied. If 'l1', each ring is divided by the sum of all rings for that sample.",
    ),
]

type RingFractionType = Annotated[
    int,
    Field(
        description="Fraction of the rings to be used for training. If 2, takes the first half of the rings for each layer. If 3, takes the first third of the rings, and so on.",
    ),
]


class HistoryTrainMetrics(TypedDict):
    loss: list[float]
    accuracy: list[float]


class HistoryValMetrics(TypedDict):
    loss: list[float]
    accuracy: list[float]
    max_sp: list[float]
    max_sp_pd: list[float]
    max_sp_fa: list[float]
    max_sp_threshold: list[float]
    max_sp_best_epoch: list[int]


class HistoryDict(FitRoutineDict):
    train: HistoryTrainMetrics
    val: HistoryValMetrics


class FitTrainMetrics(TypedDict):
    loss: float
    accuracy: float


class FitValMetrics(TypedDict):
    loss: float
    accuracy: float
    max_sp: float
    max_sp_pd: float
    max_sp_fa: float
    max_sp_threshold: float
    max_sp_best_epoch: int


class FitDict(TypedDict):
    train: FitTrainMetrics
    val: FitValMetrics


# class EvaluationDict(EnhancedConfusionMatrixDict):
#     loss: float
#     weighted: NotRequired[EnhancedConfusionMatrixDict]


class TrainingResult(TypedDict):
    fold: int
    init: int
    et_bin: BinDict
    eta_bin: AbsoluteBinDict
    history: HistoryDict
    fit: FitDict
    # train: EvaluationDict
    # val: EvaluationDict
    # test: EvaluationDict


type PatienceType = Annotated[
    int,
    Field(
        description="Number of epochs with no improvement after which training will be stopped.",
    ),
]


def validate_selection_criteria_key(value: str) -> str:
    valid_keys_list = [
        f"fit.train.{metric}" for metric in FitTrainMetrics.__annotations__.keys()
    ] + [f"fit.val.{metric}" for metric in FitValMetrics.__annotations__.keys()]
    if value not in valid_keys_list:
        raise ValueError(
            f"Invalid selection criteria key: {value}. Must be one of {valid_keys_list}"
        )
    return value


type SelectionCriteriaKeyType = Annotated[
    str,
    Field(
        description="Criterion to select the best model initilization or fold groups. Must be a key in the fit dictionary of the training results.",
    ),
    AfterValidator(validate_selection_criteria_key),
]


class SelectionCriteria(BaseModel):
    key: SelectionCriteriaKeyType
    criterion: Literal["max", "min"] = Field(
        "max",
        description="How to select the best model from initilization or fold groups. Must be either 'max' or 'min'.",
    )

    def select_best_expr(self) -> pl.Expr:
        if self.criterion == "max":
            return pl.all().sort_by(self.key, descending=True).first()
        elif self.criterion == "min":
            return pl.all().sort_by(self.key, descending=False).first()
        else:
            raise ValueError(
                f"Invalid criterion: {self.criterion}. Must be either 'max' or 'min'."
            )


type EtaBinIntervalValue = Annotated[
    float,
    Field(
        ge=0,
        description="Represents the value of the Eta bin interval. Must be a non-negative integer.",
    ),
]


class BinValidator:
    def __init__(self, bin_class: type[Bin]):
        self.bin_class = bin_class

    def __call__(self, et_bins: list[Bin | float]) -> list[Bin]:
        length = len(et_bins)
        is_float = isinstance(et_bins[0], float)
        if length < 2 and is_float:
            raise ValueError(
                f"et_bins must have at least 2 values to define a bin. Got {length} value(s)."
            )
        if is_float:
            et_bins = [
                self.bin_class(low=et_bins[i], high=et_bins[i + 1], closed="left")
                for i in range(length - 1)
            ]
            return et_bins
        return et_bins


type QuantizerConfigType = "QuantizerConfig" | None


class PreprocessingPipeline:
    def __init__(
        self,
        ring_selector: RingSlicesPerLayer,
        normalizer: AlternativeNorm1 | None = None,
    ):
        self.ring_selector = ring_selector
        self.normalizer = normalizer

    @classmethod
    def from_job_params(
        cls,
        rings_col: RingsColType,
        ring_fraction: RingFractionType = 2,
        norm_strategy: NormStrategyType = None,
    ):
        ring_selector = RingSlicesPerLayer(
            rings_col=rings_col,
            fraction=ring_fraction,
            output_format="expanded_columns",
        )
        if norm_strategy == "l1":
            normalizer = AlternativeNorm1(
                input_cols=ring_selector.output_cols,
            )
        elif norm_strategy is None:
            return None
        else:
            raise ValueError(f"Unsupported norm_strategy: {norm_strategy}")
        return cls(
            ring_selector=ring_selector,
            normalizer=normalizer,
        )

    @property
    def input_cols(self) -> list[str]:
        return self.ring_selector.input_cols

    @property
    def output_cols(self) -> list[str]:
        if self.normalizer is None:
            return self.ring_selector.output_cols

        return self.normalizer.output_cols

    @cached_property
    def aux_cols(self) -> list[str]:
        if self.normalizer is None:
            return []
        return self.ring_selector.output_cols

    def __call__(
        self, data: pl.DataFrame | pl.LazyFrame, passthrough: bool = False
    ) -> pl.DataFrame | pl.LazyFrame:
        data = data.pipe(self.ring_selector, passthrough=passthrough)
        if self.normalizer:
            data = data.pipe(self.normalizer, passthrough=passthrough)
        return data


class BinDict(TypedDict):
    low: float
    high: float
    closed: Literal["left", "right", "both", "neither"]


class InferencePipeline:
    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline,
        committee: BinnedCommittee,
        eta_col: EtaColType,
        et_col: EtColType,
    ):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.committee = committee
        self.eta_col = eta_col
        self.et_col = et_col

    @staticmethod
    def get_abs_eta_col_name(eta_col: EtaColType) -> str:
        return f"{eta_col}_abs"

    @cached_property
    def input_cols(self) -> list[str]:
        return self.preprocessing_pipeline.input_cols + [self.eta_col, self.et_col]

    @cached_property
    def output_cols(self) -> list[str]:
        return self.committee.output_cols

    def __call__(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        batch_size: int = 32,
        all_layers: bool = False,
        passthrough: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        return (
            data.pipe(self.preprocessing_pipeline, passthrough=True)
            .with_columns(
                pl.col(self.eta_col)
                .abs()
                .alias(self.get_abs_eta_col_name(self.eta_col))
            )
            .pipe(
                self.committee.predict_polars,
                all_layers=all_layers,
                passthrough=passthrough,
                batch_size=batch_size,
            )
        )

    @classmethod
    def from_job_params(
        cls,
        rings_col: RingsColType,
        ring_fraction: RingFractionType,
        norm_strategy: NormStrategyType,
        eta_col: EtaColType,
        et_col: EtColType,
        keras_models: list["Sequential"],
        et_bins: list[BinDict],
        eta_bins: list[BinDict],
        decision_thresholds: list[float] | None = None,
    ) -> Self:
        preprocessing_pipeline = PreprocessingPipeline.from_job_params(
            rings_col=rings_col,
            ring_fraction=ring_fraction,
            norm_strategy=norm_strategy,
        )

        if decision_thresholds is None:
            decision_thresholds = (None for _ in range(len(keras_models)))

        abs_eta_col_name = cls.get_abs_eta_col_name(eta_col)
        binned_models = []
        iterator = zip(et_bins, eta_bins, keras_models, decision_thresholds)
        for et_bin, eta_bin, keras_model, decision_threshold in iterator:
            binned_model = BinnedModel(
                bins=[
                    VariableBin(var_name=et_col, **et_bin),
                    VariableBin(var_name=abs_eta_col_name, **eta_bin),
                ],
                keras_model=keras_model,
                features=preprocessing_pipeline.output_cols,
                decision_threshold=decision_threshold,
            )
            binned_models.append(binned_model)

        committee = BinnedCommittee(
            models=binned_models,
        )
        return cls(
            preprocessing_pipeline=preprocessing_pipeline,
            committee=committee,
            eta_col=eta_col,
            et_col=et_col,
        )


class MLPKerasTrainingJob(YamlBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Dataset params
    batch_size: BatchSizeType = 32
    data_table: DataTableType
    dataset_dir: DirectoryType
    et_col: EtColType
    et_bins: EtBinType
    eta_col: EtaColType
    eta_bins: EtaBinType
    fold_col: FoldColType
    kfold_table: KFoldTableType
    label_col: LabelColType
    norm_strategy: NormStrategyType = None
    rings_col: RingsColType
    ring_fraction: RingFractionType = 2

    # Model params
    model_factory: MLPFactory = Field(
        ...,
        description="Description of the MLP model to train.",
    )
    from_logits: FromLogitsType = False
    epochs: EpochsType = 5000
    verbose: VerboseType = 1
    loss: LossType
    optimizer: OptimizerType
    balance_class_weights: BalanceClassWeightsType = True
    inits: Annotated[int, Field(description="Number of initializations")] = 5
    patience: PatienceType = 25

    # Threshold Fit
    # num_thresholds: int = Field(
    #     ge=2,
    #     description="Number of thresholds to evaluate for the strategy.",
    # )
    # lower_threshold: float = Field(
    #     ge=0,
    #     lt=1,
    #     description="Lower threshold limit for the strategy.",
    # )
    # upper_threshold: float = Field(
    #     gt=0,
    #     le=1,
    #     description="Upper threshold limit for the strategy.",
    # )

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
    logger_name: LoggerName = None

    # Model Selection
    best_init: SelectionCriteria = Field(
        default_factory=SelectionCriteria,
        description="Criterion to select the best model initialization. Must be a key in the fit dictionary of the training results.",
    )
    best_fold: SelectionCriteria = Field(
        default_factory=SelectionCriteria,
        description="Criterion to select the best fold. Must be a key in the fit dictionary of the training results.",
    )

    # # Private
    # _thresholds: list[float] | None = PrivateAttr(None)

    # @property
    # def thresholds(self) -> list[float]:
    #     if self._thresholds is None:
    #         raise ValueError("Thresholds not initialized")
    #     return self._thresholds

    # @cached_property
    # def thresholds_array(self) -> npt.NDArray[np.floating]:
    #     return np.array(self.thresholds)

    @cached_property
    def config_path(self) -> Path:
        return self.output_path / "config.json"

    @cached_property
    def all_models_results_path(self) -> Path:
        return self.output_path / "all_models_results.parquet"

    @cached_property
    def selected_models_path(self) -> Path:
        return self.output_path / "selected_models.parquet"

    @cached_property
    def preprocessing_pipeline(self) -> PreprocessingPipeline:
        return PreprocessingPipeline.from_job_params(
            ring_fraction=self.ring_fraction,
            rings_col=self.rings_col,
            norm_strategy=self.norm_strategy,
        )

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

    def get_member_output_dir(self, member_id: int) -> Path:
        return self.output_path / f"member_{member_id}"

    def get_member_model_path(self, member: Path | int) -> Path:
        if isinstance(member, int):
            member = self.get_member_output_dir(member)
        return member / "model.keras"

    def get_member_results_path(self, member: Path | int) -> Path:
        if isinstance(member, int):
            member = self.get_member_output_dir(member)
        return member / "results.json.zip"

    # def model_post_init(self, context):
    #     res = super().model_post_init(context)
    #     if self.lower_threshold >= self.upper_threshold:
    #         raise ValueError(
    #             f"lower_threshold must be less than upper_threshold. Got lower_threshold={self.lower_threshold} and upper_threshold={self.upper_threshold}."
    #         )
    #     self._thresholds = np.linspace(
    #         self.lower_threshold,
    #         self.upper_threshold,
    #         self.num_thresholds,
    #     )

    #     self._thresholds = (
    #         self._thresholds.tolist()
    #     )  # Convert to list for JSON serialization

    #     return res

    def get_dataset(
        self, fold: int, et_bin: BinDict, eta_bin: BinDict, **kwargs
    ) -> RingerParquetDataset:
        config_dict = dict(
            dataset_dir=self.dataset_dir,
            data_table=self.data_table,
            rings_col=self.rings_col,
            kfold_table=self.kfold_table,
            label_col=self.label_col,
            fold_col=self.fold_col,
            fold=fold,
            et_bin={"var_name": self.et_col, **et_bin},
            eta_bin={"var_name": self.eta_col, **eta_bin},
        )
        config_dict.update(kwargs)
        dataset = RingerParquetDataset(**config_dict)
        return dataset

    def get_numpy_data(self, df: pl.LazyFrame) -> tuple[np.ndarray, np.ndarray]:
        df = df.pipe(self.preprocessing_pipeline, passthrough=True)
        X = df.select(self.preprocessing_pipeline.output_cols)
        y = df.select(self.label_col)
        X, y = pl.collect_all([X, y])
        X = X.to_numpy()
        y = y.to_numpy().flatten()
        return X, y

    def get_inference_pipeline(
        self,
        et_col: str | None = None,
        eta_col: str | None = None,
        rings_col: str | None = None,
    ) -> InferencePipeline:

        if et_col is None:
            et_col = self.et_col

        if eta_col is None:
            eta_col = self.eta_col

        if rings_col is None:
            rings_col = self.rings_col

        keras_models = []
        et_bins = []
        eta_bins = []

        for row in self.selected_models.iter_rows(named=True):
            member_id = row["id"]
            keras_model: "Sequential" = self.get_member_model(member_id)
            keras_models.append(keras_model)
            et_bins.append(
                {
                    "low": row["et_bin.low"],
                    "high": row["et_bin.high"],
                    "closed": row["et_bin.closed"],
                }
            )
            eta_bins.append(
                {
                    "low": row["eta_bin.low"],
                    "high": row["eta_bin.high"],
                    "closed": row["eta_bin.closed"],
                }
            )

        inference_pipeline = InferencePipeline.from_job_params(
            rings_col=rings_col,
            ring_fraction=self.ring_fraction,
            norm_strategy=self.norm_strategy,
            eta_col=eta_col,
            et_col=et_col,
            keras_models=keras_models,
            et_bins=et_bins,
            eta_bins=eta_bins,
        )

        return inference_pipeline

    def submit(self):

        logger = logging.getLogger(self.logger_name)
        logger.info(
            f"Starting job with model {self.model_factory.name} on dataset at {self.dataset_dir}"
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_path.joinpath("config.json").write_text(
            self.model_dump_json(indent=4),
            encoding="utf-8",
        )

        dataset = RingerParquetDataset(
            dataset_dir=self.dataset_dir,
            data_table=self.data_table,
            rings_col=self.rings_col,
            kfold_table=self.kfold_table,
            label_col=self.label_col,
            fold_col=self.fold_col,
            fold=0,
        )

        n_folds = dataset.get_n_folds()
        logger.info(f"The dataset has {n_folds} folds.")

        folds_range = range(n_folds)
        inits_range = range(self.inits)
        iterator = product(self.et_bins, self.eta_bins, folds_range, inits_range)
        executor = self.executor_config.get_executor()
        submitted_jobs = []
        with executor.batch():
            for member_id, (et_bin, eta_bin, fold, init) in enumerate(iterator):
                member_output_path = self.get_member_output_dir(member_id)
                try:
                    MLPKerasTrainingJob.validate_saved_member_directory(
                        member_output_path
                    )
                    logger.info(
                        f"{member_id}: Member directory {member_output_path} already exists and is valid. Skipping training."
                    )
                    continue
                except (FileNotFoundError, NotADirectoryError):
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
        dependent_executor.submit(self.post_training)

        logger.info("All jobs submitted.")

    def _run_training(
        self, member_id: int, et_bin: dict, eta_bin: dict, fold: int, init: int
    ):
        from neuralnet.models.keras.routines import fit_routine
        from keras import Model

        logger = logging.getLogger(self.logger_name)
        logger.info(
            f"{member_id}: Running training for Et bin {et_bin} and Eta bin {eta_bin}, fold {fold} and init {init}"
        )
        dataset = self.get_dataset(
            fold=fold,
            et_bin=et_bin,
            eta_bin=eta_bin,
        )

        results: TrainingResult = {
            "fold": fold,
            "init": init,
            "et_bin": et_bin,
            "eta_bin": eta_bin,
        }
        output_dir = self.get_member_output_dir(member_id)
        output_dir.mkdir(parents=True, exist_ok=True)
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

        train_data = self.get_numpy_data(dataset.train_df())
        val_data = self.get_numpy_data(dataset.val_df())

        sp_callback = SP(
            validation_data=val_data,
            patience=self.patience,
            verbose=self.verbose,
            save_the_best=True,
            from_logits=self.from_logits,
        )
        callbacks = [sp_callback]
        model, results["history"] = fit_routine(
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
            class_weight=class_weights,
        )
        logger.info(f"Finished training for fold {fold} and init {init}")

        model_path = self.get_member_model_path(output_dir)
        model.save(str(model_path))

        best_epoch = results["history"]["val"]["max_sp_best_epoch"][-1]
        results["fit"] = {
            "train": {
                metric_name: metric_history[best_epoch]
                for metric_name, metric_history in results["history"]["train"].items()
            },
            "val": {
                metric_name: metric_history[best_epoch]
                for metric_name, metric_history in results["history"]["val"].items()
            },
        }

        # results["train"] = self.run_evaluation(
        #     model=model,
        #     data=train_numpy,
        #     class_weight=dataset.train_class_weights()
        #     if self.balance_class_weights
        #     else None,
        # )
        # del train_numpy

        # results["val"] = self.run_evaluation(
        #     model=model,
        #     data=val_numpy,
        #     class_weight=dataset.val_class_weights()
        #     if self.balance_class_weights
        #     else None,
        # )
        # del val_numpy

        # if not self.dry_run and hasattr(dataset, "test_df"):
        #     test_numpy = self.get_numpy_data(dataset.test_df())
        #     results["test"] = self.run_evaluation(
        #         model=model,
        #         data=test_numpy,
        #         class_weight=dataset.test_class_weights()
        #         if self.balance_class_weights
        #         else None,
        #     )
        #     del test_numpy

        logger.info(f"Finished evaluating for fold {fold} and init {init}")
        zip_path = self.get_member_results_path(output_dir)
        results_path = Path(str(zip_path).replace(".zip", ""))
        from ...json import cast_to_json_value

        with results_path.open("w", encoding="utf-8") as f:
            json.dump(cast_to_json_value(results), f, indent=4)

        with ZipFile(zip_path, "w", ZIP_DEFLATED) as zipf:
            zipf.write(results_path, arcname="results.json")

        logger.info(f"Saved results to {output_dir}")

        del model
        results_path.unlink()
        from keras.backend import clear_session

        clear_session()

    # def run_evaluation(
    #     self,
    #     model: "Model",
    #     data: tuple[np.ndarray, np.ndarray],
    #     class_weight: dict[int, float] | None = None,
    # ) -> EvaluationDict:

    #     predictions = model.predict(
    #         data[0], batch_size=self.batch_size, verbose=self.verbose
    #     )
    #     if self.from_logits:
    #         predictions = sigmoid(predictions)
    #     eval_dict = enhanced_confusion_matrix_from_preds(data[1], predictions)
    #     if class_weight:
    #         weighted_eval_dict = {
    #             "tn": np.array(eval_dict["tn"]) * class_weight[0],
    #             "tp": np.array(eval_dict["tp"]) * class_weight[1],
    #             "fn": np.array(eval_dict["fn"]) * class_weight[1],
    #             "fp": np.array(eval_dict["fp"]) * class_weight[0],
    #         }
    #         weighted_enhanced_cm_dict = enhanced_confusion_matrix(
    #             tn=weighted_eval_dict["tn"],
    #             tp=weighted_eval_dict["tp"],
    #             fn=weighted_eval_dict["fn"],
    #             fp=weighted_eval_dict["fp"],
    #             thresholds=np.array(eval_dict["thresholds"]),
    #         )
    #         eval_dict["weighted"] = weighted_enhanced_cm_dict
    #     return eval_dict

    def post_training(self):
        logger = logging.getLogger(self.logger_name)
        all_models_results = defaultdict(list)
        for member_path in self.output_path.glob("member_*"):
            member_id = int(member_path.name.split("_")[-1])
            all_models_results["id"].append(member_id)
            results_path = self.get_member_results_path(member_path)
            with ZipFile(results_path, "r") as zipf:
                with zipf.open("results.json") as f:
                    member_results = json.load(f)

            for key, value in traverse(member_results, include_sequences=False):
                all_models_results[key].append(value)

        logger.info(
            f"Computing best models based on the selection criteria: init: {self.best_init}, fold: {self.best_fold}"
        )
        all_model_results_df = pl.DataFrame(all_models_results).with_columns(
            pl.col("history.start").str.to_datetime(), pl.col("history.end").str.to_datetime()
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
        best_init_results = all_model_results_df.group_by(*bins_cols, "fold").agg(
            self.best_init.select_best_expr()
        )
        selected_models = best_init_results.group_by(*bins_cols).agg(
            self.best_fold.select_best_expr()
        )
        selected_models.write_parquet(self.selected_models_path)

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
            MLPKerasTrainingJob.validate_saved_member_directory(member_output_path)

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
        results_path = member_output_path / "results.json.zip"
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found in {member_output_path}.")

    @classmethod
    def load(cls, path: Path | str) -> Self:
        cls.validate_saved_directory(path)
        path = Path(path)
        config_path = path / "config.json"
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        instance = cls(**config)
        return instance

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

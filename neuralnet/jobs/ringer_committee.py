from typing import Annotated, NotRequired, Self, TypedDict
from itertools import product
from dataclasses import dataclass
from pydantic import Field, ConfigDict, PrivateAttr, BaseModel
import logging
from pathlib import Path
import typer
import numpy as np
import numpy.typing as npt
from functools import cached_property
from ..pydantic import YamlBaseModel
from ..submitit import ExecutorConfig
from ..logging import LoggerName
from ..datasets.ringer import (
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
from ..utils import pydantic_to_markdown_schema
from ..numpy import inverse_sigmoid
from ..metrics import enhanced_confusion_matrix, EnhancedConfusionMatrixDict
from ..models.keras.factories import (
    EpochsType,
    VerboseType,
    StandardFitDict,
    KerasSequentialModelFactory,
    LossType,
    OptimizerType,
)

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


class FitDict(StandardFitDict):
    loss: list[float]
    accuracy: list[float]
    max_sp_val: list[float]
    max_sp_pd_val: list[float]
    max_sp_fa_val: list[float]
    max_sp_threshold_val: list[float]


class EvaluationDict(EnhancedConfusionMatrixDict):
    loss: float
    weighted: NotRequired[EnhancedConfusionMatrixDict]


class KerasBinaryClassificationJobResults(TypedDict):
    fold: int
    init: int
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


class NoDatasetFieldRingerTrainingJob(BaseModel):
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


class RingerTrainingJob(YamlBaseModel, NoDatasetFieldRingerTrainingJob):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: RingerTrainingJobDatasetType
    _thresholds: npt.NDArray[np.floating] | None = PrivateAttr(default=None)

    @property
    def thresholds(self) -> npt.NDArray[np.floating]:
        if self._thresholds is None:
            raise ValueError("Thresholds not initialized")
        return self._thresholds

    @cached_property
    def thresholds_list(self) -> list[float]:
        return self.thresholds.tolist()

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

        return res

    def submit(self, executor=None):
        logger = logging.getLogger(self.logger_name)
        logger.info(
            f"Starting job with model {self.model.name} on dataset at {self.dataset.dataset_dir}"
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_path.joinpath("config.json").write_text(
            self.model_dump_json(indent=4, exclude={"models", "results"}),
            encoding="utf-8",
        )
        n_folds = self.dataset.get_n_folds()
        logger.info(f"The dataset has {n_folds} folds.")
        folds_range = range(n_folds)
        inits_range = range(self.inits)
        fold_init_iterator = enumerate(product(folds_range, inits_range))
        if executor is None:
            executor = self.executor_config.get_executor()
            with executor.batch():
                self._submit(executor, fold_init_iterator)
        else:
            self._submit(executor, fold_init_iterator)
        logger.info("All training jobs submitted.")

    def _submit(self, executor, fold_init_iterator):
        logger = logging.getLogger(self.logger_name)
        for i, (fold, init) in fold_init_iterator:
            logger.info(f"Submitting training for fold {fold} and init {init}")
            executor.submit(
                self._run_training,
                fold=fold,
                init=init,
            )
            if i > 0 and self.dry_run:
                logger.info("Dry run enabled, stopping after first iteration.")
                break
            logger.info(
                f"{i} - Submitting training job for fold {fold} and init {init}"
            )

    def _run_training(self, fold: int, init: int):
        from neuralnet.models.keras.routines import fit_routine, evaluation_routine
        from neuralnet.tensorflow.callbacks import SP
        from keras import Model

        logger = logging.getLogger(self.logger_name)
        logger.info(f"Running training for fold {fold} and init {init}")
        self.dataset.set_fold(fold)
        train_numpy = self.dataset.train_numpy()
        val_numpy = self.dataset.val_numpy()
        results: KerasBinaryClassificationJobResults = {
            "fold": fold,
            "init": init,
        }
        model_factory = self.model_factory.model_copy(
            update=dict(name=f"{self.model_factory.name}_fold_{fold}_init_{init}")
        )
        model: Model = model_factory.as_keras()
        if self.balance_class_weights:
            class_weights = self.dataset.train_class_weights()
        else:
            class_weights = None

        if self.dry_run:
            logger.warning("Dry run enabled, running one epoch.")
            epochs = 1
        else:
            epochs = model_factory.epochs

        sp_callback = SP(
            validation_data=val_numpy,
            patience=self.patience,
            verbose=self.verbose,
            save_the_best=True,
        )
        callbacks = [c.as_keras() for c in model_factory.callbacks] + [sp_callback]

        results["fit"] = fit_routine(
            model=model,
            dataset=self.dataset,
            loss=self.loss.as_keras(),
            optimizer=self.optimizer.as_keras(),
            metrics=["accuracy"],
            callbacks=callbacks,
            epochs=epochs,
            verbose=self.verbose,
            class_weights=class_weights,
        )
        logger.info(f"Finished training for fold {fold} and init {init}")

        eval_dict = evaluation_routine(
            model=model,
            dataset=train_numpy,
            loss=model_factory.loss.as_keras(),
            optimizer=model_factory.optimizer.as_keras(),
            metrics=self.get_metrics(),
        )
        results["train"] = self.process_eval_dict(eval_dict, dataset_type="train")
        del train_numpy

        if not self.dry_run and hasattr(self.dataset, "val_numpy"):
            val_numpy = self.dataset.val_numpy()
            eval_dict = evaluation_routine(
                model=model,
                dataset=val_numpy,
                loss=model_factory.loss.as_keras(),
                optimizer=model_factory.optimizer.as_keras(),
                metrics=self.get_metrics(),
            )
            results["val"] = self.process_eval_dict(eval_dict, dataset_type="val")
            del val_numpy

        if not self.dry_run and hasattr(self.dataset, "test_numpy"):
            test_numpy = self.dataset.test_numpy()
            eval_dict = evaluation_routine(
                model=model,
                dataset=test_numpy,
                loss=model_factory.loss.as_keras(),
                optimizer=model_factory.optimizer.as_keras(),
                metrics=self.get_metrics(),
            )
            results["test"] = self.process_eval_dict(eval_dict, dataset_type="test")
            del test_numpy
        logger.info(f"Finished evaluating for fold {fold} and init {init}")
        output_path = self.output_path / f"fold_{fold}_init_{init}"
        model.save(output_path)

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
                thresholds=self.thresholds_list,
            ),
            TruePositives(
                name="true_positives",
                thresholds=self.thresholds_list,
            ),
            FalseNegatives(
                name="false_negatives",
                thresholds=self.thresholds_list,
            ),
            FalsePositives(
                name="false_positives",
                thresholds=self.thresholds_list,
            ),
        ]
        return metrics

    def process_eval_dict(
        self,
        eval_dict: dict,
        dataset_type: str,
    ) -> EvaluationDict:
        import numpy as np

        enhanced_cm_dict = enhanced_confusion_matrix(
            tn=np.array(eval_dict.pop("true_negatives")),
            tp=np.array(eval_dict.pop("true_positives")),
            fn=np.array(eval_dict.pop("false_negatives")),
            fp=np.array(eval_dict.pop("false_positives")),
            thresholds=self.thresholds_list,
        )
        if self.balance_class_weights:
            match dataset_type:
                case "train":
                    class_weights = self.dataset.train_class_weights()
                case "val":
                    class_weights = self.dataset.val_class_weights()
                case "test":
                    class_weights = self.dataset.test_class_weights()
                case _:
                    raise ValueError(f"Unknown dataset_type: {dataset_type}")
            weighted_eval_dict = {
                "true_negatives": np.array(eval_dict["true_negatives"])
                * class_weights[0],
                "true_positives": np.array(eval_dict["true_positives"])
                * class_weights[1],
                "false_negatives": np.array(eval_dict["false_negatives"])
                * class_weights[0],
                "false_positives": np.array(eval_dict["false_positives"])
                * class_weights[1],
            }
            weighted_enhanced_cm_dict = enhanced_confusion_matrix(
                tn=weighted_eval_dict["true_negatives"],
                tp=weighted_eval_dict["true_positives"],
                fn=weighted_eval_dict["false_negatives"],
                fp=weighted_eval_dict["false_positives"],
                thresholds=self.thresholds_list,
            )
            enhanced_cm_dict["weighted"] = weighted_enhanced_cm_dict
        return enhanced_cm_dict


@dataclass
class BinaryClassificationJobResults:
    @classmethod
    def from_zip(cls) -> Self:
        raise NotImplementedError(
            "Loading BinaryClassificationJobResults from zip is not implemented yet."
        )


class RingerCommitteeTrainingJob(NoDatasetFieldRingerTrainingJob):
    batch_size: BatchSizeType
    data_table: DataTableType
    rings_col: RingsColType
    kfold_table: KFoldTableType
    label_col: LabelColType
    fold_col: FoldColType
    et_col: EtColType
    et_bins: list[float] = Field(
        ...,
        description="Bins to be used for the Et variable. Must be a list of increasing values.",
        min_length=2
    )
    eta_col: EtaColType
    eta_bins: list[float] = Field(
        ...,
        description="Bins to be used for the Eta variable. Must be a list of increasing values.",
        ge=0,
        min_length=2
    )
    ring_fraction: RingFractionType
    norm_strategy: NormStrategyType

    def model_post_init(self, context):
        res = super().model_post_init(context)
        self.et_bins.sort()
        self.eta_bins.sort()
        return res

    def submit(self):
        import polars as pl
        et_bin_idxs = range(len(self.et_bins) - 1)
        eta_bin_idxs = range(len(self.eta_bins) - 1)
        executor = self.executor_config.get_executor()
        members = {
            'et_bin.lower': [],
            'et_bin.upper': [],
            'et_bin.closed': [],
            'eta_bin.lower': [],
            'eta_bin.upper': [],
            'eta_bin.closed': [],
            'output_path': [],
        }
        with executor.batch():
            for member_id, (et_bin_idx, eta_bin_idx) in enumerate(product(et_bin_idxs, eta_bin_idxs)):
                et_bin = Bin(
                    lower=self.et_bins[et_bin_idx],
                    upper=self.et_bins[et_bin_idx + 1],
                    closed="left",
                )
                eta_bin = Bin(
                    lower=self.eta_bins[eta_bin_idx],
                    upper=self.eta_bins[eta_bin_idx + 1],
                    closed="left",
                )
                output_path = self._submit_for_bin(et_bin, eta_bin, executor, member_id)
                members['et_bin.lower'].append(et_bin.lower)
                members['et_bin.upper'].append(et_bin.upper)
                members['et_bin.closed'].append(et_bin.closed)
                members['eta_bin.lower'].append(eta_bin.lower)
                members['eta_bin.upper'].append(eta_bin.upper)
                members['eta_bin.closed'].append(eta_bin.closed)
                members['output_path'].append(output_path)
        members_df = pl.DataFrame(members)
        members_df.write_csv(self.output_path / "members.csv", index=False, include_header=True)

    def _submit_for_bin(
        self,
        et_bin: Bin,
        eta_bin: Bin,
        executor,
        member_id: int
    ):
        logger = logging.getLogger(self.logger_name)
        logger.info(f"Submitting training for Et bin {et_bin} and Eta bin {eta_bin}")
        output_path = self.output_path / f"committee_member_{member_id}"
        job = RingerTrainingJob(
            model_factory=self.model_factory,
            from_logits=self.from_logits,
            epochs=self.epochs,
            verbose=self.verbose,
            loss=self.loss,
            optimizer=self.optimizer,
            balance_class_weights=self.balance_class_weights,
            inits=self.inits,
            num_thresholds=self.num_thresholds,
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            dry_run=self.dry_run,
            executor_config=self.executor_config,
            output_path=output_path,
            logger_name=self.logger_name,
            dataset=RingerParquetDataset(
                data_table=self.data_table,
                rings_col=self.rings_col,
                kfold_table=self.kfold_table,
                label_col=self.label_col,
                fold_col=self.fold_col,
                et_col=self.et_col,
                et_bin=et_bin,
                eta_col=self.eta_col,
                eta_bin=eta_bin,
                ring_fraction=self.ring_fraction,
                norm_strategy=self.norm_strategy,
            ),
        )
        job.submit(executor=executor)
        return output_path


app = typer.Typer(help="NeuralNet Binary Classification", rich_markup_mode="markdown")


RUN_TRAINING_HELP = "Run Ringer Training jobs"


@app.command(
    short_help=RUN_TRAINING_HELP,
    help=f"**{RUN_TRAINING_HELP}**\n\n{pydantic_to_markdown_schema(RingerTrainingJob)}",
)
def run_training(
    config: Annotated[
        Path,
        typer.Option(
            "--config", help="Path to the YAML configuration file for the training job"
        ),
    ],
):
    job = RingerTrainingJob.from_yaml(config)
    job.submit()

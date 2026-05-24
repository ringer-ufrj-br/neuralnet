from functools import cached_property
from typing import Annotated, Any
from itertools import product
from keras.models import Sequential, load_model
from keras.metrics import (
    Accuracy,
    TrueNegatives,
    TruePositives,
    FalseNegatives,
    FalsePositives,
    Recall,
)
from keras.callbacks import History, Callback
from pydantic import BaseModel, Field, PrivateAttr, JsonValue, ConfigDict
import numpy as np
import numpy.typing as npt
import logging
from datetime import datetime
from pathlib import Path
import zipfile
from zipfile import ZipFile
from io import BytesIO
import json
from dataclasses import dataclass
import polars as pl

from ..tensorflow.callbacks import SP
from ..pydantic import YamlBaseModel
from ..datasets.ringer import RingerParquetDataset
from ..submitit import ExecutorConfig
from ..numpy import inverse_sigmoid
from ..optimizers import AdamOptimizerConfig
from ..losses import BinaryCrossEntropyLossConfig
from ..logging import LoggerName
from .mlp import DenseLayerConfig
from .quantum import (
    BasicEntanglerQuantumLayerConfig,
    StronglyEntanglingQuantumLayerConfig,
    HardwareEfficientQuantumLayerConfig,
)


type LayerConfigType = Annotated[
    (
        DenseLayerConfig
        | BasicEntanglerQuantumLayerConfig
        | StronglyEntanglingQuantumLayerConfig
        | HardwareEfficientQuantumLayerConfig
    ),
    Field(discriminator="kind", description="Layer configuration field."),
]

type OptimizerConfigType = Annotated[
    AdamOptimizerConfig,
    Field(discriminator="kind", description="Optimizer configuration field."),
]

type LossConfigType = Annotated[
    BinaryCrossEntropyLossConfig,
    Field(discriminator="kind", description="Loss configuration field."),
]

type UpperThresholdLimit = Annotated[
    float,
    Field(description="Upper threshold limit for the metric."),
]

type LowerThresholdLimit = Annotated[
    float,
    Field(description="Lower threshold limit for the metric."),
]

type NumThresholds = Annotated[
    int,
    Field(
        gt=0,
        description="Number of thresholds for the metric.",
    ),
]

type FromLogitsType = Annotated[
    bool,
    Field(
        False,
        description="Whether the model output to consider is logits. When enabled considers the threshold limits as probabilities between 0 and 1",
    ),
]

type ModelDatasetType = Annotated[
    (
        tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]
        | tuple[pl.LazyFrame, pl.LazyFrame]
        | tuple[npt.NDArray[np.floating], pl.LazyFrame]
        | tuple[pl.LazyFrame, npt.NDArray[np.integer]]
    ),
    Field(description="Validation dataset as a tuple of (X_val, y_val)"),
]

type ParentArchiveType = tuple[ZipFile, str]


@dataclass(frozen=True)
class ConfusionMatrixResults:
    tn: npt.NDArray[np.integer]
    tp: npt.NDArray[np.integer]
    fn: npt.NDArray[np.integer]
    fp: npt.NDArray[np.integer]
    thresholds: npt.NDArray[np.floating]

    @cached_property
    def total(self) -> npt.NDArray[np.integer]:
        return self.tn + self.tp + self.fn + self.fp

    @cached_property
    def correct(self) -> npt.NDArray[np.integer]:
        return self.tn + self.tp

    @cached_property
    def incorrect(self) -> npt.NDArray[np.integer]:
        return self.fn + self.fp

    @cached_property
    def positives(self) -> npt.NDArray[np.integer]:
        return self.tp + self.fn

    @cached_property
    def negatives(self) -> npt.NDArray[np.integer]:
        return self.tn + self.fp

    @cached_property
    def accuracy(self) -> npt.NDArray[np.floating]:
        return self.correct / self.total if self.total > 0 else np.nan

    @cached_property
    def recall(self) -> npt.NDArray[np.floating]:
        return self.tp / self.positives if self.positives > 0 else np.nan

    @property
    def pd(self) -> npt.NDArray[np.floating]:
        return self.recall

    @cached_property
    def fpr(self) -> npt.NDArray[np.floating]:
        return self.fp / self.negatives if self.negatives > 0 else np.nan

    @cached_property
    def fa(self) -> npt.NDArray[np.floating]:
        return self.fpr

    @cached_property
    def sp_index(self) -> npt.NDArray[np.floating]:
        """Calculate the SP index for each threshold."""
        return np.sqrt(
            np.sqrt(self.pd * (1 - self.fa)) * (0.5 * (self.pd + (1 - self.fa)))
        )

    @cached_property
    def auc(self) -> float:
        """Calculate the AUC using the trapezoidal rule."""
        # Sort by FPR
        sorted_indices = np.argsort(self.fpr)
        sorted_fpr = self.fpr[sorted_indices]
        sorted_pd = self.pd[sorted_indices]
        return np.trapz(sorted_pd, sorted_fpr)


class BinaryClassificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field("keras_sequential", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    layers: list[LayerConfigType] = Field(
        ..., min_items=1, description="List of sequential model layers."
    )

    loss: LossConfigType
    optimizer: OptimizerConfigType
    from_logits: FromLogitsType
    num_thresholds: NumThresholds
    lower_threshold: LowerThresholdLimit
    upper_threshold: UpperThresholdLimit

    # Training related fields
    epochs: int = Field(5000, description="Number of epochs to train the model.", gt=0)
    batch_size: int = Field(32, gt=0, description="Batch size to train the model.")
    verbose: int = Field(1, ge=0, le=2, description="Verbosity of the training.")
    patience: int = Field(
        25,
        description="Number of epochs with no improvement after which training will be stopped.",
    )

    # Misc
    logger_name: LoggerName

    _keras: Sequential | None = PrivateAttr(default=None)
    _fit_history: JsonValue | None = PrivateAttr(default=None)
    _thresholds: npt.NDArray[np.floating] | None = PrivateAttr(default=None)

    def _compile(self):
        self._keras.compile(
            optimizer=self.optimizer.get(),
            loss=self.loss.get(),
            metrics=self.get_metrics(),
        )

    def model_post_init(self, __context):
        if self.lower_threshold >= self.upper_threshold:
            raise ValueError("Lower threshold must be lower than upper threshold")
        if self.from_logits:
            if self.lower_threshold < 0 or self.upper_threshold > 1:
                raise ValueError(
                    "Thresholds must be between 0 and 1 when from_logits is True"
                )

        self._keras = Sequential([layer.get() for layer in self.layers], name=self.name)
        self._compile()
        metric_thresholds = np.linspace(
            start=self.lower_threshold,
            stop=self.upper_threshold,
            num=self.num_thresholds,
        )
        if self.from_logits:
            metric_thresholds = inverse_sigmoid(metric_thresholds)
        self._thresholds = metric_thresholds

    @property
    def thresholds(self) -> npt.NDArray[np.floating]:
        if self._thresholds is None:
            raise ValueError("Thresholds not initialized")
        return self._thresholds

    @cached_property
    def thresholds_list(self) -> list[float]:
        return self.thresholds.tolist()

    @property
    def keras(self) -> Sequential:
        return self._keras

    @property
    def fit_history(self) -> JsonValue | None:
        return self._fit_history

    def set_keras(self, keras_model: Sequential, fit_history: JsonValue | None = None):
        self._keras = keras_model
        self._compile()
        self._fit_history = fit_history

    def get_metrics(
        self,
    ) -> tuple[
        Accuracy, TrueNegatives, TruePositives, FalseNegatives, FalsePositives, Recall
    ]:
        """Return a tuple of metrics to be used in the model."""
        return (
            Accuracy(),
            TrueNegatives(thresholds=self.thresholds_list),
            TruePositives(thresholds=self.thresholds_list),
            FalseNegatives(thresholds=self.thresholds_list),
            FalsePositives(thresholds=self.thresholds_list),
            Recall(thresholds=self.thresholds_list),
        )

    def get_callbacks(self, val_dataset: ModelDatasetType) -> tuple[SP]:
        return [
            SP(
                validation_data=val_dataset,
                patience=25,
                verbose=self.verbose,
                save_the_best=True,
            )
        ]

    def fit(
        self,
        train_dataset: ModelDatasetType,
        val_dataset: ModelDatasetType,
        callbacks: list[Callback],
    ) -> History:
        logger = logging.getLogger(self.logger_name)
        callbacks = callbacks + self.get_callbacks(val_dataset)
        start = datetime.now()
        history = self._keras.fit(
            *train_dataset,
            validation_data=val_dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=callbacks,
            shuffle=True,
        )
        end = datetime.now()
        logger.info(f"Finished training for model {self.name} with history: {history}")
        logger.info(f"Training step: {end - start}")
        self._fit_history = history.history

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._keras.predict(X)

    def evaluate(
        self, dataset: tuple[np.ndarray, np.ndarray]
    ) -> ConfusionMatrixResults:
        _, tn, tp, fn, fp, _ = self._keras.evaluate(*dataset, verbose=self.verbose)
        results = ConfusionMatrixResults(
            tn=tn,
            tp=tp,
            fn=fn,
            fp=fp,
            thresholds=self.thresholds,
        )

        return results

    def _save_to_zip(self, archive: ZipFile, parent_dir: str):
        if parent_dir and not parent_dir.endswith("/"):
            parent_dir += "/"
        model_buffer = BytesIO()
        self._keras.save(model_buffer)
        archive.writestr(f"{parent_dir}keras_model.keras", model_buffer.getvalue())
        model_json = self.model_dump_json(indent=4)
        archive.writestr(f"{parent_dir}model_config.json", model_json)
        if self._fit_history is not None:
            history_json = json.dumps(self._fit_history, indent=4)
            archive.writestr(f"{parent_dir}fit_history.json", history_json)

    def save(
        self,
        path: Path | str,
        compression: int = zipfile.ZIP_DEFLATED,
        parent_archive: ParentArchiveType | None = None,
    ):
        if isinstance(path, str):
            path = Path(path)
        if path.exists():
            raise FileExistsError(
                f"Path {path} already exists. Please provide a non existing path to save the model."
            )

        if parent_archive is None:
            with zipfile.ZipFile(path, mode="w", compression=compression) as archive:
                self._save_to_zip(archive, "")
        else:
            self._save_to_zip(parent_archive[0], parent_archive[1])

    @classmethod
    def load(cls, path: Path | str, base_dir: str = "", custom_objects: Any = None):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist. Cannot load model.")

        if base_dir and not base_dir.endswith("/"):
            base_dir += "/"

        with zipfile.ZipFile(path, mode="r") as archive:
            with archive.open(f"{base_dir}model_config.json") as config_file:
                config_dict = json.load(config_file)
                model = cls(**config_dict)
            with archive.open(f"{base_dir}keras_model.keras") as model_file:
                keras_model = load_model(model_file, custom_objects=custom_objects)
            try:
                with archive.open(f"{base_dir}fit_history.json") as history_file:
                    fit_history = json.load(history_file)
            except KeyError:
                fit_history = None  # fit history is optional

        model.set_keras(keras_model, fit_history)
        return model


type BinaryClassificationJobDatasetType = Annotated[
    RingerParquetDataset,
    Field(
        description="Validation dataset as a RingerParquetDataset instance.",
        discriminator="kind",
    ),
]


class BinaryClassificationJob(YamlBaseModel):
    dataset: BinaryClassificationJobDatasetType
    model: BinaryClassificationModel

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
    inits: Annotated[int, Field(description="Number of initializations")] = 5
    output_path: Annotated[
        Path, Field(description="Path to save the results of the job")
    ]

    # Misc
    logger_name: LoggerName

    def run(self):
        logger = logging.getLogger(self.logger_name)
        logger.info(
            f"Starting job with model {self.model.name} on dataset at {self.dataset.dataset_dir}"
        )

        n_folds = self.dataset.get_n_folds()
        logger.info(f"The dataset has {n_folds} folds.")
        folds_range = range(n_folds)
        inits_range = range(self.inits)
        executor = self.executor_config.get_executor()
        for i, (fold, init) in enumerate(product(folds_range, inits_range)):
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

        logger.info("All training jobs submitted.")
    
    def _run_training(self, fold: int, init: int):
        logger = logging.getLogger(self.logger_name)
        logger.info(f"Running training for fold {fold} and init {init}")
        train_dataset, val_dataset = self.dataset.get_fold_data(fold)
        self.model.fit(train_dataset, val_dataset, callbacks=[])
        results = self.model.evaluate(val_dataset)
        logger.info(f"Finished training for fold {fold} and init {init} with results: {results}")
        output_dir = self.output_path / f"fold_{fold}" / f"init_{init}"
        output_dir.mkdir(parents=True, exist_ok=False)
        self.model.save(output_dir / "model.zip")
        with open(output_dir / "results.json", "w") as f:
            json.dump(
                {
                    "confusion_matrix": {
                        "tn": results.tn.tolist(),
                        "tp": results.tp.tolist(),
                        "fn": results.fn.tolist(),
                        "fp": results.fp.tolist(),
                    },
                    "thresholds": results.thresholds.tolist(),
                    "accuracy": results.accuracy.tolist(),
                    "recall": results.recall.tolist(),
                    "fpr": results.fpr.tolist(),
                    "sp_index": results.sp_index.tolist(),
                    "auc": results.auc,
                },
                f,
                indent=4,
            )

    def save_complete(self, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        if path.exists():
            raise FileExistsError(
                f"Path {path} already exists. Please provide a non existing path to save the job."
            )
        with ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            job_config = self.model_dump_json(indent=4)
            archive.writestr("job_config.json", job_config)
            self.model.save("", parent_archive=(archive, "model/"))

from functools import cached_property
from typing import Annotated, Any, TypedDict
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
from keras.callbacks import Callback, History
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
import numpy as np
import numpy.typing as npt
import logging
from datetime import datetime
from pathlib import Path
import polars as pl
import json
import typer

from ..tensorflow.callbacks import SP
from ..pydantic import YamlBaseModel
from ..datasets.ringer import RingerParquetDataset
from ..submitit import ExecutorConfig
from ..numpy import inverse_sigmoid
from ..optimizers import AdamOptimizer
from ..losses import BinaryCrossEntropyLossConfig
from ..logging import LoggerName
from ..metrics import ConfusionMatrix, ConfusionMatrixDict
from ..utils import pydantic_to_markdown_schema
from .mlp import DenseLayer
from .quantum import (
    BasicEntanglerQuantumLayer,
    StronglyEntanglingQuantumLayer,
    HardwareEfficientQuantumLayer,
)


type LayerType = Annotated[
    (
        DenseLayer
        | BasicEntanglerQuantumLayer
        | StronglyEntanglingQuantumLayer
        | HardwareEfficientQuantumLayer
    ),
    Field(discriminator="name", description="Layer configuration field."),
]

type OptimizerConfigType = Annotated[
    AdamOptimizer,
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
    tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]],
    Field(description="Validation dataset as a tuple of (X_val, y_val)"),
]

class BinaryClassificationModelHistory(TypedDict):
    loss: list[float]
    accuracy: list[float]
    true_negatives: list[npt.NDArray[np.floating]]
    true_positives: list[npt.NDArray[np.floating]]
    false_negatives: list[npt.NDArray[np.floating]]
    false_positives: list[npt.NDArray[np.floating]]
    recall: list[npt.NDArray[np.floating]]
    max_sp_val: list[float]
    max_sp_pd_val: list[float]
    max_sp_best_epoch: list[int]


class BinaryClassificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field("keras_sequential", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    layers: list[LayerType] = Field(
        ..., min_items=1, description="List of sequential model layers."
    )

    loss: LossConfigType
    optimizer: OptimizerConfigType
    from_logits: FromLogitsType
    num_thresholds: NumThresholds
    lower_threshold: LowerThresholdLimit
    upper_threshold: UpperThresholdLimit
    # thresholds_strategy: Annotated[
    #     Literal['max_sp'],
    #     Field(
    #         'max_sp',
    #         description="Strategy to select the best threshold. Currently only supports 'max_sp' which selects the threshold with the highest SP index.",
    #     ),
    # ]

    # Training related fields
    epochs: int = Field(5000, description="Number of epochs to train the model.", gt=0)

    verbose: int = Field(1, ge=0, le=2, description="Verbosity of the training.")
    patience: int = Field(
        25,
        description="Number of epochs with no improvement after which training will be stopped.",
    )

    # Misc
    logger_name: LoggerName

    _keras: Sequential | None = PrivateAttr(default=None)
    _thresholds: npt.NDArray[np.floating] | None = PrivateAttr(default=None)

    def _compile(self):
        self._keras.compile(
            optimizer=self.optimizer.get(),
            loss=self.loss.get(),
            metrics=list(self.get_metrics()),
        )

    def model_post_init(self, __context):
        if self.lower_threshold >= self.upper_threshold:
            raise ValueError("Lower threshold must be lower than upper threshold")
        if self.from_logits:
            if self.lower_threshold < 0 or self.upper_threshold > 1:
                raise ValueError(
                    "Thresholds must be between 0 and 1 when from_logits is True"
                )

        metric_thresholds = np.linspace(
            start=self.lower_threshold,
            stop=self.upper_threshold,
            num=self.num_thresholds,
        )
        if self.from_logits:
            metric_thresholds = inverse_sigmoid(metric_thresholds)
        self._thresholds = metric_thresholds
        
        self._keras = Sequential([layer.get() for layer in self.layers], name=self.name)
        self._compile()

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

    def set_keras(self, keras_model: Sequential):
        self._keras = keras_model
        self._compile()

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
    
    def _clean_history(self, history: History) -> BinaryClassificationModelHistory:
        new_history = {}
        for key, value in history.history.items():
            splitted_key = key.split('_')
            try:
                float(splitted_key[-1])
                new_key = '_'.join(splitted_key[:-1])
                new_history[new_key] = value
            except (ValueError, IndexError):
                new_history[key] = value
        return new_history

    def fit(
        self,
        train_dataset: ModelDatasetType,
        val_dataset: ModelDatasetType,
        callbacks: list[Callback],
    ) -> BinaryClassificationModelHistory:
        logger = logging.getLogger(self.logger_name)
        callbacks = callbacks + self.get_callbacks(val_dataset)
        start = datetime.now()
        history = self._keras.fit(
            *train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=callbacks,
            shuffle=True,
        )
        end = datetime.now()
        logger.info(f"Finished training for model {self.name} with history: {history}")
        logger.info(f"Training step: {end - start}")

        history = self._clean_history(history)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._keras.predict(X)

    def evaluate(self, dataset: tuple[np.ndarray, np.ndarray]) -> ConfusionMatrix:
        loss, _, tn, tp, fn, fp, _ = self._keras.evaluate(*dataset, verbose=self.verbose)
        results = ConfusionMatrix(
            loss=loss,
            tn=tn,
            tp=tp,
            fn=fn,
            fp=fp,
            thresholds=self.thresholds,
        )

        return results

    def save(
        self,
        path: Path | str
    ):
        if isinstance(path, str):
            path = Path(path)
        if path.exists():
            raise FileExistsError(f"Path {path} already exists. Cannot save model.")
        if path.is_file():
            raise FileExistsError(f"Path {path} is a file. Cannot save model.")
        
        path.mkdir(parents=True, exist_ok=False)
        self._keras.save(path.joinpath("model.keras"))
        with path.joinpath("config.json").open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=4)

    @classmethod
    def load(cls, path: Path | str, custom_objects: Any = None):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist. Cannot load model.")
        if path.is_file():
            raise FileNotFoundError(f"Path {path} is a file. Cannot load model.")
        
        with path.joinpath("config.json").open("r", encoding="utf-8") as f:
            config_dict = json.load(f)
            model = cls(**config_dict)
        keras_model = load_model(path.joinpath("model.keras"), custom_objects=custom_objects, safe_mode=False)
        model.set_keras(keras_model)
        return model
    
    def new(self, **kwargs) -> "BinaryClassificationModel":
        """Create a new instance of the model with the same configuration but a new Keras model."""
        config_dict = self.model_dump()
        config_dict.update(kwargs)
        try:
            return BinaryClassificationModel(**config_dict)
        finally:
            model = self._keras
            self._keras = None
            del model


type BinaryClassificationJobDatasetType = Annotated[
    RingerParquetDataset,
    Field(
        description="Validation dataset as a RingerParquetDataset instance.",
        discriminator="kind",
    ),
]

class BinaryClassificationModelResultsDict(TypedDict):
    fold: int
    init: int
    fit: BinaryClassificationModelHistory
    train: ConfusionMatrixDict
    val: ConfusionMatrixDict
    test: ConfusionMatrixDict


class BinaryClassificationJob(YamlBaseModel):

    model_config = ConfigDict(extra="forbid", frozen=True,
                              arbitrary_types_allowed=True)

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

    models: list[BinaryClassificationModel] = Field(
        default_factory=list,
        description="List of models to be trained. This field is populated when the job is loaded.",
    )
    results: pl.DataFrame | None = Field(
        default=None,
        description="DataFrame to store the results of the job. This field is populated after the job is run."
    )

    def submit(self):
        logger = logging.getLogger(self.logger_name)
        logger.info(
            f"Starting job with model {self.model.name} on dataset at {self.dataset.dataset_dir}"
        )
        self.output_path.mkdir(parents=True, exist_ok=False)
        self.output_path.joinpath("config.json").write_text(
            self.model_dump_json(indent=4, exclude={'models', 'results'}), encoding="utf-8"
        )
        n_folds = self.dataset.get_n_folds()
        logger.info(f"The dataset has {n_folds} folds.")
        folds_range = range(n_folds)
        inits_range = range(self.inits)
        executor = self.executor_config.get_executor()
        with executor.batch():
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
        self.dataset.set_fold(fold)
        train_numpy = self.dataset.train_numpy()
        val_numpy = self.dataset.val_numpy()
        results: BinaryClassificationModelResultsDict = {
            'fold': fold,
            'init': init,
        }
        model = self.model.new(name=f"{self.model.name}_fold_{fold}_init_{init}")
        results["fit"] = model.fit(train_numpy, val_numpy, callbacks=[])
        logger.info(f"Finished training for fold {fold} and init {init}")
        results["train"] = model.evaluate(train_numpy).to_dict()
        del train_numpy
        results["val"] = model.evaluate(val_numpy).to_dict()
        del val_numpy
        test_numpy = self.dataset.test_numpy()
        results["test"] = model.evaluate(test_numpy).to_dict()
        del test_numpy
        logger.info(f"Finished evaluating for fold {fold} and init {init}")
        output_path = self.output_path / f"fold_{fold}_init_{init}"
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                return super().default(obj)
        
        model.save(output_path)
        with output_path.joinpath("results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

    @classmethod
    def load(cls, path: Path | str):
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist. Cannot load job.")
        if path.is_file():
            raise FileNotFoundError(f"Path {path} is a file. Cannot load job.")

        with path.joinpath("config.json").open('r') as f:
            job_config = json.load(f)

        job_config['models'] = []
        results = []

        for model_dir in path.glob("fold_*_init_*"):
            if model_dir.is_file():
                raise FileNotFoundError(f"Expected {model_dir} to be a directory containing the model and results, but it is a file.")
            model = BinaryClassificationModel.load(model_dir)
            job_config["models"].append(model)
            with model_dir.joinpath("results.json").open('r') as f:
                res = json.load(f)
                results.append(res)
        job_config['results'] = pl.from_dicts(results)
        job = cls(**job_config)
        return job


app = typer.Typer(help="NeuralNet Binary Classification", rich_markup_mode="markdown")


RUN_TRAINING_HELP = "Run Binary Classification jobs"


@app.command(
    short_help=RUN_TRAINING_HELP,
    help=f"**{RUN_TRAINING_HELP}**\n\n{pydantic_to_markdown_schema(BinaryClassificationJob)}",
)
def run_training(
    config: Annotated[
        Path,
        typer.Option(
            "--config", help="Path to the YAML configuration file for the training job"
        ),
    ],
):
    job = BinaryClassificationJob.from_yaml(config)
    job.run()

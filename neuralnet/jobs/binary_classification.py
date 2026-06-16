from typing import Annotated, Self, TypedDict
from itertools import product
from dataclasses import dataclass
from pydantic import Field, ConfigDict
import numpy as np
import logging
from pathlib import Path
import typer
from ..pydantic import YamlBaseModel
from ..submitit import ExecutorConfig
from ..logging import LoggerName
from ..models.keras.binary_classification import (
    BinaryClassificationModel,
    BinaryClassificationModelResultsDict,
)
from ..datasets.ringer import RingerParquetDataset
from ..utils import pydantic_to_markdown_schema
from ..models.binary_classification import (
    FitDict,
    EvaluationDict,
    KerasSequentialBinaryClassificationModelFactory,
)

type BinaryClassificationJobDatasetType = Annotated[
    RingerParquetDataset,
    Field(
        description="Dataset to be used.",
        discriminator="object_type",
    ),
]


class KerasSequentialBinaryClassificationModelResultsDict(TypedDict):
    fold: int
    init: int
    fit: FitDict
    train: EvaluationDict
    val: EvaluationDict
    test: EvaluationDict


class BinaryClassificationJob(YamlBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: BinaryClassificationJobDatasetType
    model: KerasSequentialBinaryClassificationModelFactory

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

    def submit(self):
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
        from ..models.keras.binary_classification import KerasSequentialBinaryClassificationModel
        logger = logging.getLogger(self.logger_name)
        logger.info(f"Running training for fold {fold} and init {init}")
        self.dataset.set_fold(fold)
        train_numpy = self.dataset.train_numpy()
        val_numpy = self.dataset.val_numpy()
        results: KerasSequentialBinaryClassificationModelResultsDict = {
            "fold": fold,
            "init": init,
        }
        model: KerasSequentialBinaryClassificationModel = self.model.model_copy(update=dict(
            name=f"{self.model.name}_fold_{fold}_init_{init}"
        )).as_keras()
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
        model.save(output_path)


@dataclass
class BinaryClassificationJobResults:
    @classmethod
    def from_zip(cls) -> Self:
        raise NotImplementedError(
            "Loading BinaryClassificationJobResults from zip is not implemented yet."
        )


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
    job.submit()

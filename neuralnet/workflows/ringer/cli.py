import typer
from typing import Annotated
from pathlib import Path
from ...pydantic import pydantic_to_markdown_schema
from .training import RingerKerasTrainingJob
from .threshold_fit import RingerCommitteeThresholdFitJob
from .alternative_norm1 import AlternativeNorm1Analysis


app = typer.Typer(help="Ringer Committee command line interface", rich_markup_mode="markdown")


RUN_TRAINING_HELP = "Run Ringer Committtee Trigger Training jobs"


@app.command(
    short_help=RUN_TRAINING_HELP,
    help=(
        f"{RUN_TRAINING_HELP}\n\n"
        "This command runs the training jobs for the Ringer Committee Trigger. "
        "The configuration for the training jobs is provided through a YAML file. "
        "The YAML file should follow the schema bellow:\n\n"
        f"{pydantic_to_markdown_schema(RingerKerasTrainingJob)}"
    ),
)
def training(
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to the YAML configuration file for the training job"),
    ],
):

    job = RingerKerasTrainingJob.from_yaml(config)
    job.submit()


RUN_THRESHOLD_FIT_HELP = "Run Ringer Committee threshold fit job"


@app.command(
    name="threshold-fit",
    short_help=RUN_THRESHOLD_FIT_HELP,
    help=(
        f"{RUN_THRESHOLD_FIT_HELP}\n\n"
        "This command evaluates a trained Ringer Committee by running inference "
        "on a (potentially different) dataset and computing metrics at configurable "
        "reference operating points.  "
        "The configuration is provided through a YAML file. "
        "The YAML file should follow the schema bellow:\n\n"
        f"{pydantic_to_markdown_schema(RingerCommitteeThresholdFitJob)}"
    ),
)
def threshold_fit(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the YAML configuration file for the threshold fit job",
        ),
    ],
):
    job = RingerCommitteeThresholdFitJob.from_yaml(config)
    job.submit()


RUN_ALTERNATIVE_NORM1_HELP = "Run AlternativeNorm1 quantization analysis"


@app.command(
    name="alternative-norm1",
    short_help=RUN_ALTERNATIVE_NORM1_HELP,
    help=(
        f"{RUN_ALTERNATIVE_NORM1_HELP}\n\n"
        "This command analyzes the impact of fixed-point quantization on AlternativeNorm1 "
        "by computing MSE, MAE, MAPE, and KL divergence across configurable bit ranges.\n\n"
        f"{pydantic_to_markdown_schema(AlternativeNorm1Analysis)}"
    ),
)
def alternative_norm1(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the YAML configuration file for the AlternativeNorm1 analysis",
        ),
    ],
):
    job = AlternativeNorm1Analysis.from_yaml(config)
    job.submit()

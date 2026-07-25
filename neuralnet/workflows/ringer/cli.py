import typer
from typing import Annotated
from pathlib import Path
from ...pydantic import pydantic_to_markdown_schema
from .training import RingerKerasTrainingJob
from .inference import InferenceJob, PTQConversionJob, FixedPointInferenceJob


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
        typer.Option(
            "--config", help="Path to the YAML configuration file for the training job"
        ),
    ],
):

    job = RingerKerasTrainingJob.from_yaml(config)
    job.submit()


@app.command(
    short_help="Run inference job",
    help=(
        "Run inference job for the Ringer Committee Trigger for MLPs. "
        "The configuration for the inference job is provided through a YAML file. "
        "The YAML file should follow the schema bellow:\n\n"
        f"{pydantic_to_markdown_schema(InferenceJob)}"
    ),
)
def inference(
    config: Annotated[
        Path,
        typer.Option(
            "--config", help="Path to the YAML configuration file for the inference job"
        ),
    ],
):
    job = InferenceJob.from_yaml(config)
    job.submit()


@app.command(
    short_help="Run fixed-point inference job",
    help=(
        "Run fixed-point inference job for the Ringer Committee Trigger for MLPs. "
        "The configuration for the inference job is provided through a YAML file. "
        "The YAML file should follow the schema bellow:\n\n"
        f"{pydantic_to_markdown_schema(FixedPointInferenceJob)}"
    ),
)
def fixed_point_inference(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the YAML configuration file for the fixed-point inference job",
        ),
    ],
):
    job = FixedPointInferenceJob.from_yaml(config)
    job.submit()


@app.command(
    short_help="Run PTQ conversion job",
    help=(
        "Run PTQ conversion job for the Ringer Committee Trigger for MLPs. "
        "The configuration for the job is provided through a YAML file. "
        "The YAML file should follow the schema bellow:\n\n"
        f"{pydantic_to_markdown_schema(PTQConversionJob)}"
    ),
)
def ptq_conversion(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the YAML configuration file for the job",
        ),
    ],
):
    job = PTQConversionJob.from_yaml(config)
    job.submit()

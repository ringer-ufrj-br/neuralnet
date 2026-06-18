import typer
from typing import Annotated
from pathlib import Path
from ...pydantic import pydantic_to_markdown_schema
from .jobs import MLPKerasTrainingJob


app = typer.Typer(
    help="MLP ringer command line interface", rich_markup_mode="markdown"
)


RUN_TRAINING_HELP = "Run Ringer Committtee Trigger Training jobs"


@app.command(
    short_help=RUN_TRAINING_HELP,
    help=(
        f"{RUN_TRAINING_HELP}\n\n"
        "This command runs the training jobs for the Ringer Committee Trigger for MLPs. "
        "The configuration for the training jobs is provided through a YAML file. "
        "The YAML file should follow the schema bellow:\n\n"
        f"{pydantic_to_markdown_schema(MLPKerasTrainingJob)}"
    ),
)
def run_training(
    config: Annotated[
        Path,
        typer.Option(
            "--config", help="Path to the YAML configuration file for the training job"
        ),
    ],
):

    job = MLPKerasTrainingJob.from_yaml(config)
    job.submit()

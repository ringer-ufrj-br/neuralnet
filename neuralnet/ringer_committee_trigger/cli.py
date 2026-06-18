import typer
from typing import Annotated
from pathlib import Path


app = typer.Typer(
    help="Ringer Committee Trigger command line interface", rich_markup_mode="markdown"
)


RUN_TRAINING_HELP = "Run Ringer Training jobs"


@app.command(
    short_help="Run Ringer Committtee Trigger Training jobs",
)
def run_training(
    config: Annotated[
        Path,
        typer.Option(
            "--config", help="Path to the YAML configuration file for the training job"
        ),
    ],
):
    from neuralnet.ringer_committee_trigger.keras_training_job import RingerTrainingJob

    job = RingerTrainingJob.from_yaml(config)
    job.submit()

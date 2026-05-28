import typer
# from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file before importing other modules that might rely on them
from neuralnet.datasets import app as dataset_app
from neuralnet.models.vkan import app as vkan_app
from neuralnet.models.vqat import app as vqat_app
from neuralnet.models.mlp import app as mlp_app
from neuralnet.models.binary_classification import app as binary_classification_app

# load_dotenv(Path(__file__).parent.joinpath(".env"))  # Load environment variables from .env file
# import os
# print(os.getenv("KERAS_BACKEND"))

app = typer.Typer(help="Ringer Zero CLI", rich_markup_mode="markdown")
app.add_typer(vqat_app, name="vqat")
app.add_typer(vkan_app, name="vkan")
app.add_typer(mlp_app, name="mlp")
app.add_typer(dataset_app, name="datasets")
app.add_typer(binary_classification_app, name="binary-classification")

if __name__ == "__main__":
    app()

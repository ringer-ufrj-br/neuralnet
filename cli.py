import typer
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file before importing other modules that might rely on them
# from neuralnet.models.vkan import app as vkan_app
from neuralnet.workflows.ringer.cli import app as ringer_app


app = typer.Typer(help="NeuralNet CLI", rich_markup_mode="markdown")
app.add_typer(ringer_app, name="ringer")
# app.add_typer(vkan_app, name="kan")

if __name__ == "__main__":
    app()

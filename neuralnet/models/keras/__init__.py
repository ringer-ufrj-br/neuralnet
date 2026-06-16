from abc import abstractmethod, ABC
from typing import Self, TypedDict, Any, NotRequired, Generator
from pydantic import JsonValue
from keras import Loss, Model, Optimizer, Metric, Layer
from keras.models import load_model
from keras.callbacks import Callback, History
from datetime import datetime
import logging
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from tempfile import TemporaryDirectory
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, asdict

from ...json import cast_to_json_value

from ..keras_factories import (
    NameType,
    JitCompileType,
    EpochsType,
    VerboseType,
    PatienceType,
    KerasModelFactory,
    KerasSequentialModelFactory,
    StandardFitDict,
    StandardEvaluationDict
)


class CompileDict(TypedDict):
    loss: Loss
    optimizer: Optimizer
    jit_compile: JitCompileType
    metrics: NotRequired[list[Metric]]


type KerasData = tuple[
    npt.NDArray[np.floating | np.integer], npt.NDArray[np.floating | np.integer]
]

type KerasModelType = Model | None


@dataclass
class KerasModel(ABC):
    name: NameType
    callbacks: list[Callback]
    loss: Loss
    optimizer: Optimizer
    jit_compile: JitCompileType

    # Training related fields
    epochs: EpochsType
    verbose: VerboseType
    patience: PatienceType

    keras_model: KerasModelType = None

    @abstractmethod
    def get_model(self) -> Model:
        pass

    def fit_compile_kwargs(self) -> CompileDict:
        return {
            "loss": self.loss,
            "optimizer": self.optimizer,
            "jit_compile": self.jit_compile,
        }

    def evaluate_compile_kwargs(self) -> CompileDict:
        return {
            "loss": self.loss,
            "optimizer": self.optimizer,
            "jit_compile": self.jit_compile,
        }

    def fit(
        self,
        train_data: KerasData,
        val_data: KerasData,
        extra_callbacks: list[Callback],
        class_weight: dict[int, float] | None = None,
    ) -> StandardFitDict:
        logger = logging.getLogger("neuralnet")
        callbacks = self.callbacks + extra_callbacks
        start = datetime.now()
        if self.keras_model is None:
            self.keras_model = self.get_model()
        self.keras_model.compile(**self.compile_kwargs())
        history: History = self.keras_model.fit(
            *train_data,
            validation_data=val_data,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=callbacks,
            shuffle=True,
            class_weight=class_weight,
        )
        end = datetime.now()
        logger.info(f"Finished training for model {self.name} with history: {history}")
        logger.info(f"Training step: {end - start}")

        history: dict[str, JsonValue] = cast_to_json_value(history.history)
        history["start"] = start
        history["end"] = end

        return history

    def evaluate(self, data: KerasData) -> StandardEvaluationDict:
        logger = logging.getLogger("neuralnet")
        start = datetime.now()
        results = self.keras_model.evaluate(*data, verbose=self.verbose)
        results = cast_to_json_value(results)
        end = datetime.now()
        logger.info(
            f"Finished evaluating for model {self.name} with results: {results}"
        )
        logger.info(f"Evaluation step: {end - start}")
        results["start"] = start
        results["end"] = end

        return results

    @abstractmethod
    def predict(self, data: KerasData) -> Generator[dict[str, Any]]:
        pass

    def save(self, path: Path | str):
        """
        Saves the model to the given directory path with the following structure:
        - path
            - model.keras (the Keras model saved in the HDF5 format)
            - config.json (the model configuration saved in JSON format)

        Parameters
        ----------
        path : Path | str
            The path where the model will be saved.

        Raises
        ------
        FileExistsError
            If the path is a file and not a directory.
        """
        if isinstance(path, str):
            path = Path(path)
        if path.is_file():
            raise FileExistsError(f"Path {path} is a file. Cannot save model.")

        path.mkdir(parents=True, exist_ok=False)
        self._keras.save(path.joinpath("model.keras"))
        with path.joinpath("config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: Path | str, custom_objects: Any = None) -> Self:
        """
        Loads the model

        Parameters
        ----------
        path : Path | str
            The directory path where the model is saved. The directory should have teh same structure built by the save method.
        custom_objects : Any, optional
            Custom objects needed by keras to instantiate this model, by default None

        Returns
        -------
        KerasModel
            The loaded model instance.

        Raises
        ------
        FileNotFoundError
            If the path does not exist or is a file.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist. Cannot load model.")
        if path.is_file():
            raise FileNotFoundError(f"Path {path} is a file. Cannot load model.")

        with path.joinpath("config.json").open("r", encoding="utf-8") as f:
            config_dict = json.load(f)

        config_dict["keras_model"] = load_model(
            path.joinpath("model.keras"), custom_objects=custom_objects, safe_mode=False
        )
        model = cls(**config_dict)
        return model

    def _to_zip(self, archive: ZipFile, prefix: Path):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.save(temp_path)
            for file in temp_path.rglob("*"):
                if prefix:
                    arcname = str(Path(prefix) / file.relative_to(temp_path))
                else:
                    arcname = str(file.relative_to(temp_path))
                archive.write(file, arcname=arcname)

    def to_zip(self, archive: Path | str | ZipFile, prefix: Path | str = ""):
        """
        Saves the model to a zip archive following the same structure as the save method.
        The prefix param allows for hierarchical saving on existing archives.

        Parameters
        ----------
        archive : Path | str | ZipFile
            _description_
        prefix : Path | str, optional
            _description_, by default ''

        Raises
        ------
        FileExistsError
            _description_
        """
        if isinstance(prefix, str):
            prefix = Path(prefix)

        if isinstance(archive, str):
            archive = Path(archive)

        if isinstance(archive, ZipFile):
            self._to_zip(archive, prefix)
        else:
            if archive.exists():
                raise FileExistsError(
                    f"Archive {archive} already exists. Cannot save model."
                )
            with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zip_file:
                self._to_zip(zip_file, prefix)
    
    @classmethod
    def from_factory(cls, factory: KerasModelFactory) -> Self:
        return cls(
            name=factory.name,
            callbacks=[c.as_keras() for c in factory.callbacks],
            loss=factory.loss.as_keras(),
            optimizer=factory.optimizer.as_keras(),
            jit_compile=factory.jit_compile,
            epochs=factory.epochs,
            verbose=factory.verbose,
            patience=factory.patience,
        )


class KerasSequentialModel(KerasModel):
    layers: list[Layer]

    def get_model(self) -> Model:
        from keras import Sequential

        model = Sequential(
            [layer.as_keras() for layer in self.layers],
            name=self.name,
        )
        return model

    @classmethod
    def from_factory(cls, factory: KerasSequentialModelFactory) -> Self:
        return cls(
            name=factory.name,
            callbacks=[c.as_keras() for c in factory.callbacks],
            loss=factory.loss.as_keras(),
            optimizer=factory.optimizer.as_keras(),
            jit_compile=factory.jit_compile,
            epochs=factory.epochs,
            verbose=factory.verbose,
            patience=factory.patience,
            layers=[layer.as_keras() for layer in factory.layers],
        )

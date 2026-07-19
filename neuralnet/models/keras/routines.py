from keras import Loss, Model, Optimizer, Metric
from keras.callbacks import Callback, History
from datetime import datetime
import logging
import numpy as np
from pydantic import JsonValue

from ...json import cast_to_json_value

from .factories import (
    EpochsType,
    VerboseType,
    FitRoutineDict,
    StandardEvaluationDict,
)


def safe_jit_compile(model: Model, **compile_kwargs) -> Model:
    from keras.config import backend

    if backend() == "torch":
        try:
            compile_kwargs["jit_compile"] = True
            model.compile(**compile_kwargs)
        except Exception as e:
            compile_kwargs["jit_compile"] = False
            logger = logging.getLogger("neuralnet")
            logger.warning(f"Failed to compile model with JIT: {e}")
            model.compile(**compile_kwargs)
    else:
        model.compile(**compile_kwargs)
    return model


def fit_routine(
    model: Model,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray] | None,
    loss: Loss,
    optimizer: Optimizer,
    metrics: list[Metric | str] | None = None,
    callbacks: list[Callback] | None = None,
    class_weight: dict[int, float] | None = None,
    epochs: EpochsType = 100,
    verbose: VerboseType = 1,
    batch_size: int | None = None,
) -> tuple[Model, FitRoutineDict]:
    logger = logging.getLogger("neuralnet")
    if metrics is None:
        metrics = []

    if callbacks is None:
        callbacks = []

    start = datetime.now()
    model = safe_jit_compile(model, loss=loss, optimizer=optimizer, metrics=metrics)

    history: History = model.fit(
        *train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        shuffle=True,
        batch_size=batch_size,
        class_weight=class_weight,
    )
    end = datetime.now()
    # logger.info(f"Finished training for model {model.name} with history: {history}")
    logger.info(f"Training step: {end - start}")

    history: dict[str, JsonValue] = cast_to_json_value(history.history)
    new_history = {
        "start": start,
        "end": end,
        "train": {},
        "val": {},
    }
    for key in history.keys():
        if key.startswith("val_"):
            new_key = key.replace("val_", "")
            new_history["val"][new_key] = history[key]
        else:
            new_history["train"][key] = history[key]

    return model, new_history


def evaluation_routine(
    model: Model,
    data: tuple[np.ndarray, np.ndarray],
    loss: Loss,
    optimizer: Optimizer,
    metrics: list[Metric | str] | None = None,
    callbacks: list[Callback] | None = None,
    verbose: VerboseType = 1,
    batch_size: int | None = None,
) -> StandardEvaluationDict:
    from ... import get_logger

    logger = get_logger()
    if metrics is None:
        metrics = []

    if callbacks is None:
        callbacks = []

    start = datetime.now()
    model = safe_jit_compile(model, loss=loss, optimizer=optimizer, metrics=metrics)
    results = model.evaluate(
        *data, verbose=verbose, return_dict=True, batch_size=batch_size
    )
    results = cast_to_json_value(results)
    end = datetime.now()
    # logger.info(f"Finished evaluating for model {model.name} with results: {results}")
    logger.info(f"Evaluation step: {end - start}")
    results["start"] = start
    results["end"] = end

    return results

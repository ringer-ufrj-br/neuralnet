from typing import Any, Generator, Self

from keras.metrics import (
    Metric,
    TrueNegatives,
    TruePositives,
    FalseNegatives,
    FalsePositives,
)
from keras.losses import Loss
from keras.optimizers import Optimizer
from pydantic import Field, validate_call

from ...tensorflow.callbacks import SP

from . import KerasSequentialModel, CompileDict as BaseCompileDict
from ...numpy import inverse_sigmoid
from ...datasets.numpy import NumpyDatasetReturnTypes
from ...datasets.utils import ClassWeightedDataset
from ..binary_classification import FitDict, EvaluationDict
from ..binary_classification import (
    ThresholdStrategy,
    ModelDatasetType,
    KerasSequentialBinaryClassificationModelFactory,
    FromLogitsType,
)


class CompileDict(BaseCompileDict):
    loss: Loss
    optimizer: Optimizer
    jit_compile: bool
    metrics: list[Metric | str]


class KerasSequentialBinaryClassificationModel(KerasSequentialModel):
    """
    Model configuration and utilities for training, evaluating, saving, and loading a Keras Sequential model for binary classification tasks. The model supports custom layers, optimizers, and loss functions, as well as configurable thresholds for evaluation metrics.
    """

    from_logits: FromLogitsType
    threshold_strategy: ThresholdStrategy
    balance_class_weights: bool = Field(
        True,
        description="Whether to balance class weights during training. If True, the class weights will be set inversely proportional to the class frequencies.",
    )

    # Training related fields
    patience: int = Field(
        25,
        description="Number of epochs with no improvement after which training will be stopped.",
    )

    def model_post_init(self, context):
        res = super().model_post_init(context)
        if self.from_logits:
            self.threshold_strategy.transform_thresholds(inverse_sigmoid)
        return res

    def fit_compile_kwargs(self) -> CompileDict:
        compile_kwargs = super().fit_compile_kwargs()
        compile_kwargs["metrics"] = ["accuracy"]
        return compile_kwargs

    def evaluate_compile_kwargs(self) -> CompileDict:
        compile_kwargs = super().evaluate_compile_kwargs()
        compile_kwargs["metrics"] = [
            TrueNegatives(
                name="true_negatives",
                thresholds=self.threshold_strategy.thresholds_list,
            ),
            TruePositives(
                name="true_positives",
                thresholds=self.threshold_strategy.thresholds_list,
            ),
            FalseNegatives(
                name="false_negatives",
                thresholds=self.threshold_strategy.thresholds_list,
            ),
            FalsePositives(
                name="false_positives",
                thresholds=self.threshold_strategy.thresholds_list,
            ),
        ]
        return compile_kwargs

    @validate_call(config=dict(strict=True))
    def fit(self, dataset: ModelDatasetType) -> FitDict:

        if not isinstance(dataset, ModelDatasetType):
            raise TypeError(
                f"Expected dataset to be of type ModelDatasetType, got {type(dataset).__name__}"
            )

        val_data = dataset.val_numpy()

        if isinstance(dataset, ClassWeightedDataset):
            class_weight = dataset.class_weights()
        else:
            class_weight = None

        return super().fit(
            train_data=dataset.train_numpy(),
            val_data=val_data,
            class_weight=class_weight,
            extra_callbacks=[
                SP(
                    validation_data=val_data,
                    patience=self.patience,
                    verbose=self.verbose,
                    save_the_best=True,
                )
            ],
        )

    @validate_call(config=dict(strict=True))
    def evaluate(self, data: NumpyDatasetReturnTypes) -> EvaluationDict:
        return super().evaluate(data=data)

    def predict(
        self, dataset: ModelDatasetType, batch_size: int = 32
    ) -> Generator[dict[str, Any], None, None]:

        X, _ = dataset.test_numpy()
        prediction = self.keras_model.predict(
            X, verbose=self.verbose, batch_size=batch_size
        )
        yield {"predictions": prediction}

    @classmethod
    def from_factory(
        cls, factory: KerasSequentialBinaryClassificationModelFactory
    ) -> Self:
        return cls(
            name=factory.name,
            callbacks=[c.as_keras() for c in factory.callbacks],
            loss=factory.loss.as_keras(),
            optimizer=factory.optimizer.as_keras(),
            jit_compile=factory.jit_compile,
            layers=[layer.as_keras() for layer in factory.layers],
            from_logits=factory.from_logits,
            threshold_strategy=factory.threshold_strategy,
            balance_class_weights=factory.balance_class_weights,
            patience=factory.patience,
        )

from typing import Literal
from keras.losses import BinaryCrossentropy
from pydantic import Field

from .pydantic import ConfigModel


class BinaryCrossEntropyLossConfig(ConfigModel):
    kind: Literal["binary_cross_entropy"] = Field(
        "binary_cross_entropy",
        description='Kind of loss. Must be "binary_cross_entropy"',
    )
    from_logits: bool = Field(False, description="Whether the loss is from logits.")

    def get(self) -> BinaryCrossentropy:
        return BinaryCrossentropy(from_logits=self.from_logits)

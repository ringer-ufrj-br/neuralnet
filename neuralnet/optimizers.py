from typing import Literal
from pydantic import Field
from keras.optimizers import Adam

from .pydantic import ConfigModel


class AdamOptimizerConfig(ConfigModel):
    kind: Literal["adam"] = Field(
        "adam", description='Kind of optimizer. Must be "adam"'
    )

    learning_rate: float = Field(0.001, description="Learning rate for the optimizer.")

    def get(self) -> Adam:
        return Adam(learning_rate=self.learning_rate)

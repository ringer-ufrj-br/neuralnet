from typing import Literal
from pydantic import Field

from ..interfaces.keras import KerasFactory


class AdamFactory(KerasFactory):
    object_type: Literal["adam"] = Field("adam", description="This optimizer name.")

    learning_rate: float = Field(0.001, description="Learning rate for the optimizer.")

    def as_keras(self):
        from keras.optimizers import Adam

        return Adam(learning_rate=self.learning_rate)

from typing import Literal
from pydantic import Field, BaseModel
from keras.optimizers import Adam


class AdamOptimizer(BaseModel):
    object_type: Literal["adam"] = Field(
        "adam", description='This optimizer name.'
    )

    learning_rate: float = Field(0.001, description="Learning rate for the optimizer.")

    def get(self) -> Adam:
        return Adam(learning_rate=self.learning_rate)

from typing import Literal
from pydantic import Field, BaseModel


class BinaryCrossEntropyLossFactory(BaseModel):
    """Configuration for binary cross-entropy loss."""

    object_type: Literal["binary_cross_entropy"] = Field(
        "binary_cross_entropy",
        description='Kind of loss. Must be "binary_cross_entropy"',
    )
    from_logits: bool = Field(False, description="Whether the loss is from logits.")

    def as_keras(self):
        """
        Create a BinaryCrossentropy instance from keras configured according to the settings in this class.

        Returns
        -------
        BinaryCrossentropy
            The BinaryCrossentropy loss instance from keras
        """
        from keras.losses import BinaryCrossentropy
        return BinaryCrossentropy(from_logits=self.from_logits)

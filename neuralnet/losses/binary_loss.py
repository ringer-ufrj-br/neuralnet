"""Factory abstractions for binary cross-entropy loss configuration.

This module defines lightweight Pydantic models used by other workflows to
construct Keras binary cross-entropy loss instances from declarative settings.
"""

from typing import Literal
from pydantic import Field, BaseModel


class BinaryCrossEntropyLossFactory(BaseModel):
    """Configuration for binary cross-entropy loss.

    Attributes
    ----------
    object_type : str
        Discriminator identifying this configuration as binary cross entropy.
    from_logits : bool
        Whether the loss expects raw logits instead of probabilities.
    """

    object_type: Literal["binary_cross_entropy"] = Field(
        "binary_cross_entropy",
        description='Kind of loss. Must be "binary_cross_entropy"',
    )
    from_logits: bool = Field(False, description="Whether the loss is from logits.")

    def as_keras(self):
        """Create a configured Keras binary cross-entropy loss instance.

        Returns
        -------
        keras.losses.BinaryCrossentropy
            Keras loss object configured from this factory.
        """
        from keras.losses import BinaryCrossentropy

        return BinaryCrossentropy(from_logits=self.from_logits)

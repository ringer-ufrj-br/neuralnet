"""Factory abstractions for configuring the Adam optimizer.

This module defines a small Pydantic-backed factory used by workflows to
construct Keras Adam optimizer instances from declarative settings.
"""

from typing import Literal
from pydantic import Field, BaseModel


class AdamFactory(BaseModel):
    """Configuration for the Adam optimizer.

    Attributes
    ----------
    object_type : str
        Discriminator identifying this configuration as an Adam optimizer.
    learning_rate : float
        Learning rate passed to the Keras Adam optimizer.
    """

    object_type: Literal["adam"] = Field("adam", description="This optimizer name.")

    learning_rate: float = Field(0.001, description="Learning rate for the optimizer.")

    def as_keras(self):
        """Create a configured Keras Adam optimizer instance.

        Returns
        -------
        keras.optimizers.Adam
            Keras Adam optimizer configured from this factory.
        """
        from keras.optimizers import Adam

        return Adam(learning_rate=self.learning_rate)

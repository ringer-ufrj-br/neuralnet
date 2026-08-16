from typing import Annotated
from pydantic import Field

type RingFractionType = Annotated[
    int,
    Field(
        description="Fraction of the rings to be used for training. If 2, takes the first half of the rings for each layer. If 3, takes the first third of the rings, and so on.",
    ),
]

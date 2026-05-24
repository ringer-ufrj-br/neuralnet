from typing import Annotated
from pydantic import Field

type LoggerName = Annotated[
    str | None,
    Field(
        None,
        description="Name of the logger to be used for logging. This should correspond to a logger defined in the logging configuration.",
    ),
]

from typing import Self
from pathlib import Path
from abc import ABC
from pydantic import BaseModel, ConfigDict


class YamlBaseModel(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        import yaml

        if isinstance(path, str):
            path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

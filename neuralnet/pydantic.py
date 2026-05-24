from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel, ConfigDict


class YamlBaseModel(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, yaml_str: str):
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)


class ConfigModel(YamlBaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[""]

    @abstractmethod
    def get(self):
        pass

from typing import Protocol, runtime_checkable


@runtime_checkable
class KerasFactory(Protocol):
    def as_keras(self):
        pass

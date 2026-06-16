from typing import Protocol, runtime_checkable


@runtime_checkable
class TorchFactory(Protocol):
    def as_torch(self):
        pass

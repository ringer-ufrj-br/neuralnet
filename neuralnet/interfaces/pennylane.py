from typing import Any, Protocol, runtime_checkable, Callable


@runtime_checkable
class PennylaneFactory(Protocol):
    def as_pennylane(self):
        pass

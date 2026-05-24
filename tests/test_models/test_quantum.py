from __future__ import annotations

import importlib

import pytest

pytest.importorskip("pennylane")

from neuralnet.models.quantum import (  # noqa: E402
    BasicEntanglerQuantumLayerConfig,
    HardwareEfficientQuantumLayerConfig,
    StronglyEntanglingQuantumLayerConfig,
)


@pytest.mark.parametrize(
    ("config_cls", "expected_kind"),
    [
        (BasicEntanglerQuantumLayerConfig, "basic_entangler"),
        (StronglyEntanglingQuantumLayerConfig, "strongly_entangling"),
        (HardwareEfficientQuantumLayerConfig, "hardware_efficient"),
    ],
)
def test_quantum_layer_config_kind(config_cls, expected_kind):
    config = config_cls(n_qubits=2)
    assert config.kind == expected_kind


def _run_quantum_layer_factory(class_name: str, kwargs: dict):
    import keras

    keras.config.set_backend("torch")

    from keras.layers import TorchModuleWrapper
    from neuralnet.models import quantum

    quantum = importlib.reload(quantum)
    config_cls = getattr(quantum, class_name)
    config = config_cls(**kwargs)

    assert config.diff_method == kwargs.get("diff_method", "backprop")
    assert config.shots == kwargs.get("shots")

    layer = config.get()

    assert type(layer).__name__ == "TorchModuleWrapper"
    assert isinstance(layer, TorchModuleWrapper)
    assert str(layer.output_shape) == str(
        kwargs.get("output_shape", (None, kwargs["n_qubits"]))
    )
    assert layer.name == kwargs["name"]


@pytest.mark.parametrize(
    ("class_name", "kwargs"),
    [
        (
            "BasicEntanglerQuantumLayerConfig",
            {
                "n_qubits": 2,
                "n_layers": 2,
                "name": "basic_circuit",
                "shots": None,
                "diff_method": "backprop",
            },
        ),
        (
            "StronglyEntanglingQuantumLayerConfig",
            {
                "n_qubits": 3,
                "n_layers": 1,
                "name": "strong_circuit",
                "shots": 32,
                "diff_method": "parameter-shift",
            },
        ),
        (
            "HardwareEfficientQuantumLayerConfig",
            {
                "n_qubits": 4,
                "n_layers": 2,
                "name": "hardware_circuit",
                "output_shape": (None, 7),
                "shots": 128,
                "diff_method": "finite-diff",
            }
        ),
    ],
)
def test_quantum_layer_config_get_returns_torch_module_wrapper(
    class_name: str,
    kwargs: dict
):
    _run_quantum_layer_factory(class_name, kwargs)
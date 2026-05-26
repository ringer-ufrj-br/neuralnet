import pytest
import keras

from neuralnet.models import quantum


@pytest.mark.parametrize(
    ("class_name", "kwargs", "name"),
    [
        (
            "BasicEntanglerQuantumLayer",
            {
                "n_qubits": 2,
                "n_layers": 2,
                "name": "basic_entangler",
                "shots": None,
                "diff_method": "backprop",
            },
            "basic_entangler",
        ),
        (
            "StronglyEntanglingQuantumLayer",
            {
                "n_qubits": 3,
                "n_layers": 1,
                "name": "strongly_entangling",
                "shots": 32,
                "diff_method": "parameter-shift",
            },
            "strongly_entangling",
        ),
        (
            "HardwareEfficientQuantumLayer",
            {
                "n_qubits": 4,
                "n_layers": 2,
                "name": "hardware_efficient",
                "output_shape": (None, 7),
                "shots": 128,
                "diff_method": "finite-diff",
            },
            "hardware_efficient",
        ),
    ],
)
def test_quantum_layer_config_get_returns_torch_module_wrapper(
    class_name: str, kwargs: dict, name: str, monkeypatch: pytest.MonkeyPatch
):
    config_cls = getattr(quantum, class_name)
    config: quantum.QuantumLayer = config_cls(**kwargs)

    assert config.diff_method == kwargs.get("diff_method", "backprop")
    assert config.shots == kwargs.get("shots")
    assert config.name == name

    layer = config.get()

    assert isinstance(layer, keras.layers.TorchModuleWrapper), (
        "Expected instance of 'TorchModuleWrapper', got '{}'".format(
            type(layer).__name__
        )
    )
    assert str(layer.output_shape) == str(
        kwargs.get("output_shape", (None, kwargs["n_qubits"]))
    )
    assert layer.name == kwargs["name"]

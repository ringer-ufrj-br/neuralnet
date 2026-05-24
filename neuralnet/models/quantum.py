from typing import Literal

from pydantic import Field

from ..pydantic import ConfigModel

from keras.layers import TorchModuleWrapper


class QuantumLayerConfig(ConfigModel):
    diff_method: Literal[
        "best",
        "backprop",
        "adjoint",
        "parameter-shift",
        "finite-diff",
        "spsa",
    ] = Field(
        "backprop",
        description="PennyLane differentiation method used by the quantum node.",
    )
    n_qubits: int = Field(
        ..., gt=0, description="Number of qubits used by the quantum circuit."
    )
    n_layers: int = Field(
        2, gt=0, description="Number of trainable layers in the circuit."
    )
    device_name: str = Field(
        "default.qubit", description="PennyLane device name used to build the circuit."
    )
    shots: int | None = Field(
        default=None,
        gt=0,
        description="Number of shots used by the PennyLane device. Use None for analytic execution.",
    )
    name: str | None = Field(
        default=None, description="Optional Keras layer name for the wrapped circuit."
    )
    output_shape: tuple[int | None, ...] | None = Field(
        default=None,
        description="Optional output shape for Keras shape inference.",
    )

    def _build_torch_module_wrapper(self, ansatz, weight_shapes) -> TorchModuleWrapper:
        import pennylane as qml

        device = qml.device(
            self.device_name,
            wires=self.n_qubits,
        )

        @qml.qnode(device, interface="torch", diff_method=self.diff_method)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            ansatz(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        if self.shots is not None:
            qnode = qml.set_shots(self.shots)(qnode)

        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        return TorchModuleWrapper(
            qlayer,
            output_shape=self.output_shape or (None, self.n_qubits),
            name=self.name,
        )


class BasicEntanglerQuantumLayerConfig(QuantumLayerConfig):
    kind: Literal["basic_entangler"] = Field(
        "basic_entangler",
        description='Quantum layer kind. Uses "qml.BasicEntanglerLayers".',
    )

    def get(self) -> TorchModuleWrapper:
        import pennylane as qml

        def ansatz(weights, wires):
            qml.BasicEntanglerLayers(weights, wires=wires)

        return self._build_torch_module_wrapper(
            ansatz,
            {"weights": (self.n_layers, self.n_qubits)},
        )


class StronglyEntanglingQuantumLayerConfig(QuantumLayerConfig):
    kind: Literal["strongly_entangling"] = Field(
        "strongly_entangling",
        description='Quantum layer kind. Uses "qml.StronglyEntanglingLayers".',
    )

    def get(self) -> TorchModuleWrapper:
        import pennylane as qml

        def ansatz(weights, wires):
            qml.StronglyEntanglingLayers(weights, wires=wires)

        return self._build_torch_module_wrapper(
            ansatz,
            {"weights": (self.n_layers, self.n_qubits, 3)},
        )


class HardwareEfficientQuantumLayerConfig(QuantumLayerConfig):
    kind: Literal["hardware_efficient"] = Field(
        "hardware_efficient",
        description='Quantum layer kind. Uses a hardware-efficient Rot + CNOT ansatz.',
    )

    def get(self) -> TorchModuleWrapper:
        import pennylane as qml

        def ansatz(weights, wires):
            wire_list = list(wires)
            for layer_weights in weights:
                for wire_index, wire in enumerate(wire_list):
                    qml.Rot(
                        layer_weights[wire_index, 0],
                        layer_weights[wire_index, 1],
                        layer_weights[wire_index, 2],
                        wires=wire,
                    )
                for left_wire, right_wire in zip(wire_list, wire_list[1:]):
                    qml.CNOT(wires=[left_wire, right_wire])
                if len(wire_list) > 1:
                    qml.CNOT(wires=[wire_list[-1], wire_list[0]])

        return self._build_torch_module_wrapper(
            ansatz,
            {"weights": (self.n_layers, self.n_qubits, 3)},
        )

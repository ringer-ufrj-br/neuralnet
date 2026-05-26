from typing import Literal
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, PrivateAttr
from keras.layers import TorchModuleWrapper
from pennylane.qnn import TorchLayer
import pennylane as qml


class QuantumLayer(BaseModel, ABC):
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
    device_name: Literal["default.qubit", "lightning.qubit"] = Field(
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

    _device = PrivateAttr(None)
    _circuit_node = PrivateAttr(None)

    @staticmethod
    @abstractmethod
    def ansatz(weights, wires):
        pass

    def qnode(self, inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
        self.ansatz(weights, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def model_post_init(self, context):
        res = super().model_post_init(context)

        self._device = qml.device(
            self.device_name,
            wires=self.n_qubits,
        )

        circuit_node = qml.QNode(
            self.qnode, self._device, interface="torch", diff_method=self.diff_method
        )

        if self.shots is not None:
            circuit_node = qml.set_shots(self.shots)(circuit_node)

        self._node = circuit_node

        return res

    def _build_torch_layer(self, weight_shapes) -> TorchModuleWrapper:

        qlayer = qml.qnn.TorchLayer(self._node, weight_shapes)
        return qlayer

    def _build_torch_module_wrapper(self, ansatz, weight_shapes) -> TorchModuleWrapper:
        qlayer = self._build_torch_layer(ansatz, weight_shapes)
        return TorchModuleWrapper(
            qlayer,
            output_shape=self.output_shape or (None, self.n_qubits),
            name=self.name,
        )

    @abstractmethod
    def as_torch(self) -> TorchLayer:
        pass

    def get(self) -> TorchModuleWrapper:
        import keras

        backend = keras.config.backend()
        if backend != "torch":
            raise ValueError(
                f"Invalid Keras backend '{backend}'. Quantum layers require the 'torch' backend."
            )
        torch_layer = self.as_torch()
        return TorchModuleWrapper(
            torch_layer,
            output_shape=self.output_shape or (None, self.n_qubits),
            name=self.name,
        )


class BasicEntanglerQuantumLayer(QuantumLayer):
    name: Literal["basic_entangler"] = Field(
        "basic_entangler",
        description='Quantum layer name. Uses "qml.BasicEntanglerLayers".',
    )

    @staticmethod
    def ansatz(weights, wires):
        qml.BasicEntanglerLayers(weights, wires=wires)

    def as_torch(self) -> TorchLayer:
        return self._build_torch_layer(
            {"weights": (self.n_layers, self.n_qubits)},
        )


class StronglyEntanglingQuantumLayer(QuantumLayer):
    name: Literal["strongly_entangling"] = Field(
        "strongly_entangling",
        description='Quantum layer name. Uses "qml.StronglyEntanglingLayers".',
    )

    @staticmethod
    def ansatz(weights, wires):
        qml.StronglyEntanglingLayers(weights, wires=wires)

    def as_torch(self) -> TorchLayer:
        return self._build_torch_layer(
            {"weights": (self.n_layers, self.n_qubits, 3)},
        )


class HardwareEfficientQuantumLayer(QuantumLayer):
    name: Literal["hardware_efficient"] = Field(
        "hardware_efficient",
        description="Quantum layer name. Uses a hardware-efficient Rot + CNOT ansatz.",
    )

    @staticmethod
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

    def as_torch(self) -> TorchLayer:
        return self._build_torch_layer(
            {"weights": (self.n_layers, self.n_qubits, 3)},
        )

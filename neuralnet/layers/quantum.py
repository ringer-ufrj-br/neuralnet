from typing import Any, Literal, TypedDict, Annotated
from pydantic import Field, BaseModel, PrivateAttr

type DiffMethodType = Annotated[
    Literal["best", "backprop", "adjoint", "parameter-shift", "finite-diff", "spsa"],
    Field(
        description="PennyLane differentiation method used by the quantum node.",
    ),
]

type NameType = Annotated[
    str | None,
    Field(None, description="Optional Keras layer name for the wrapped circuit."),
]

type DeviceNameType = Annotated[
    Literal["default.qubit", "lightning.qubit"],
    Field(
        description="PennyLane device name used to build the circuit.",
    ),
]

type OutputQubitsType = Annotated[
    int | tuple[int, ...] | None,
    Field(
        description="Number of qubits or indices of qubits whose expectation values are returned as output. If None, defaults to all qubits.",
    ),
]

type ShotsType = Annotated[
    int | None,
    Field(
        gt=0,
        description="Number of shots used by the PennyLane device. Use None for analytic execution.",
    ),
]

type NQubitsType = Annotated[
    int, Field(..., gt=0, description="Number of qubits used by the quantum circuit.")
]


type AngleEmbeddingRotationType = Annotated[
    Literal["X", "Y", "Z"],
    Field(
        description="Rotation type used in the angle embedding. Can be 'X', 'Y', or 'Z'.",
    ),
]


class WeightShapesDict(TypedDict):
    weights: tuple[int, ...]


class AngleEmbedding(BaseModel):
    object_type: Literal["angle_embedding"] = Field(
        "angle_embedding",
        description='Quantum layer name. Uses "qml.AngleEmbedding".',
    )

    rotation: AngleEmbeddingRotationType = "X"

    def as_pennylane(self):
        import pennylane as qml

        rotation = self.rotation

        def embedder(inputs, wires):
            qml.AngleEmbedding(inputs, wires=wires, rotation=rotation)

        return embedder


type EmbedderType = Annotated[
    AngleEmbedding,
    Field(
        discriminator="object_type",
        description="PennyLane embedding layer used to encode the input data into the quantum circuit.",
    ),
]

type NLayersType = Annotated[
    int,
    Field(gt=0, description="Number of layers used by the quantum circuit ansatz."),
]


class BasicEntanglerAnsatz(BaseModel):
    object_type: Literal["basic_entangler_ansatz"] = Field(
        "basic_entangler_ansatz",
        description='Quantum layer name. Uses "qml.BasicEntanglerLayers".',
    )

    n_layers: NLayersType = 2

    _n_wires: int | None = PrivateAttr(None)

    @property
    def n_wires(self) -> int | None:
        return self._n_wires

    @n_wires.setter
    def n_wires(self, value: int):
        if value <= 0:
            raise ValueError("n_wires must be a positive integer.")
        self._n_wires = value

    @property
    def weight_shapes(self) -> WeightShapesDict:
        if self._n_wires is None:
            raise ValueError("n_wires must be set before accessing weight_shapes.")
        return {"weights": (self.n_layers, self.n_wires)}

    def as_pennylane(self):
        import pennylane as qml

        def ansatz(weights, wires):
            qml.BasicEntanglerLayers(weights, wires=wires)

        return ansatz


class StronglyEntanglingAnsatz(BaseModel):
    object_type: Literal["strongly_entangling_ansatz"] = Field(
        "strongly_entangling_ansatz",
        description="Quantum layer name",
    )

    n_layers: NLayersType

    _n_wires: int | None = PrivateAttr(None)

    @property
    def n_wires(self) -> int | None:
        return self._n_wires

    @n_wires.setter
    def n_wires(self, value: int):
        if value <= 0:
            raise ValueError("n_wires must be a positive integer.")
        self._n_wires = value

    @property
    def weight_shapes(self) -> WeightShapesDict:
        if self._n_wires is None:
            raise ValueError("n_wires must be set before accessing weight_shapes.")
        return {"weights": (self.n_layers, self.n_wires, 3)}

    def as_pennylane(self):
        import pennylane as qml

        def ansatz(weights, wires):
            qml.StronglyEntanglingAnsatz(weights, wires=wires)

        return ansatz


class HardwareEfficientAnsatz(BaseModel):
    object_type: Literal["hardware_efficient_ansatz"] = Field(
        "hardware_efficient_ansatz",
        description="Quantum layer name. Uses a hardware-efficient Rot + CNOT ansatz.",
    )

    n_layers: NLayersType

    _n_wires: int | None = PrivateAttr(None)

    @property
    def n_wires(self) -> int | None:
        return self._n_wires

    @n_wires.setter
    def n_wires(self, value: int):
        if value <= 0:
            raise ValueError("n_wires must be a positive integer.")
        self._n_wires = value

    @property
    def weight_shapes(self) -> WeightShapesDict:
        if self._n_wires is None:
            raise ValueError("n_wires must be set before accessing weight_shapes.")
        return {"weights": (self.n_layers, self.n_wires, 3)}

    def as_pennylane(self):
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

        return ansatz


type AnsatzType = Annotated[
    BasicEntanglerAnsatz | StronglyEntanglingAnsatz | HardwareEfficientAnsatz,
    Field(
        discriminator="object_type",
        description="PennyLane ansatz layer used to build the quantum circuit.",
    ),
]


class PauliZ(BaseModel):
    object_type: Literal["pauli_z"] = Field(
        "pauli_z",
        description='Quantum layer name. Uses "qml.BasicEntanglerLayers".',
    )

    def as_pennylane(self):
        import pennylane as qml

        def measurement(wires):
            return qml.PauliZ(wires=wires)

        return measurement


type UnaryOperatorType = Annotated[
    PauliZ,
    Field(
        discriminator="object_type",
        description="A callable that takes a single argument and returns a transformed version of it. Used to apply unary operations to the quantum circuit outputs before returning them.",
    ),
]


class QuantumLayerFactory(BaseModel):
    object_type: Literal["quantum_layer"] = Field(
        "quantum_layer",
        description='Quantum layer name. Uses "qml.BasicEntanglerLayers".',
    )

    embedder: EmbedderType = Field(default_factory=AngleEmbedding)
    ansatz: AnsatzType = Field(default_factory=BasicEntanglerAnsatz)
    unary_operator: UnaryOperatorType = Field(
        default_factory=PauliZ,
    )
    n_qubits: NQubitsType
    weight_shapes: WeightShapesDict
    device_name: DeviceNameType = "default.qubit"
    diff_method: DiffMethodType = "backprop"
    shots: ShotsType = None
    output_qubits: OutputQubitsType = None

    def _validate_output_qubits(self):
        if self.output_qubits is None:
            self.output_qubits = tuple(range(self.n_qubits))
        elif isinstance(self.output_qubits, int):
            if self.output_qubits > self.n_qubits:
                raise ValueError(
                    f"output_qubits cannot be greater than n_qubits. Got output_qubits={self.output_qubits} and n_qubits={self.n_qubits}."
                )
            self.output_qubits = (self.output_qubits,)
        elif isinstance(self.output_qubits, tuple):
            if any(q >= self.n_qubits for q in self.output_qubits):
                raise ValueError(
                    f"All qubit indices in output_qubits must be less than n_qubits. Got output_qubits={self.output_qubits} and n_qubits={self.n_qubits}."
                )
        else:
            raise ValueError(
                f"output_qubits must be an int, a tuple of ints, or None. Got {type(self.output_qubits).__name__}."
            )

    def model_post_init(self):
        self._validate_output_qubits()
        self.ansatz.n_wires = self.n_qubits

    def as_pennylane(self) -> tuple[Any, Any]:
        import pennylane as qml

        device = qml.device(
            self.device_name,
            wires=self.n_qubits,
        )
        embedder_fn = self.embedder.as_pennylane()
        ansatz_fn = self.ansatz.as_pennylane()
        unary_operator_fn = self.unary_operator.as_pennylane()

        @qml.qnode(
            device, interface="torch", diff_method=self.diff_method, shots=self.shots
        )
        def circuit_node(inputs, weights):
            embedder_fn(inputs, wires=range(self.n_qubits))
            ansatz_fn(weights, wires=range(self.n_qubits))
            return [qml.expval(unary_operator_fn(wires=i)) for i in self.output_qubits]

        return device, circuit_node

    def as_torch(self):
        from pennylane.qnn import TorchLayer

        _, circuit_node = self.as_pennylane()

        return TorchLayer(
            circuit_node,
            weight_shapes=self.ansatz.weight_shapes,
        )

    def as_keras(self):
        import keras
        from keras.layers import TorchModuleWrapper

        backend = keras.config.backend()
        if backend != "torch":
            raise ValueError(
                f"Invalid Keras backend '{backend}'. Quantum layers require the 'torch' backend."
            )
        torch_layer = self.as_torch()
        return TorchModuleWrapper(
            torch_layer,
            output_shape=self.output_shape,
            name=self.name,
        )


class BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory(QuantumLayerFactory):
    object_type: Literal["basic_entangler"] = Field(
        "basic_entangler",
        description='Quantum layer name. Uses "qml.BasicEntanglerLayers".',
    )

    ansatz: BasicEntanglerAnsatz = Field(
        default_factory=BasicEntanglerAnsatz,
        description="PennyLane ansatz layer used to build the quantum circuit. Defaults to qml.BasicEntanglerLayers.",
    )
    embedder: AngleEmbedding = Field(
        default_factory=AngleEmbedding,
        description="PennyLane embedding layer used to encode the input data into the quantum circuit. Defaults to qml.AngleEmbedding with X rotation.",
    )
    unary_operator: PauliZ = Field(
        default_factory=PauliZ,
        description="A callable that takes a single argument and returns a transformed version of it. Used to apply unary operations to the quantum circuit outputs before returning them. Defaults to qml.PauliZ measurement.",
    )


class StronglyEntanglingQuantumLayerWithAnglerEmbeddingFactory(QuantumLayerFactory):
    object_type: Literal["strongly_entangling"] = Field(
        "strongly_entangling",
        description='Quantum layer name. Uses "qml.StronglyEntanglingAnsatz".',
    )

    ansatz: StronglyEntanglingAnsatz = Field(
        default_factory=StronglyEntanglingAnsatz,
        description="PennyLane ansatz layer used to build the quantum circuit. Defaults to qml.StronglyEntanglingAnsatz.",
    )
    embedder: AngleEmbedding = Field(
        default_factory=AngleEmbedding,
        description="PennyLane embedding layer used to encode the input data into the quantum circuit. Defaults to qml.AngleEmbedding with X rotation.",
    )
    unary_operator: PauliZ = Field(
        default_factory=PauliZ,
        description="A callable that takes a single argument and returns a transformed version of it. Used to apply unary operations to the quantum circuit outputs before returning them. Defaults to qml.PauliZ measurement.",
    )


class HardwareEfficientQuantumLayerWithAnglerEmbeddingFactory(QuantumLayerFactory):
    object_type: Literal["hardware_efficient"] = Field(
        "hardware_efficient",
        description="Quantum layer name. Uses a hardware-efficient Rot + CNOT ansatz.",
    )

    ansatz: HardwareEfficientAnsatz = Field(
        default_factory=HardwareEfficientAnsatz,
        description="PennyLane ansatz layer used to build the quantum circuit. Defaults to a hardware-efficient Rot + CNOT ansatz.",
    )
    embedder: AngleEmbedding = Field(
        default_factory=AngleEmbedding,
        description="PennyLane embedding layer used to encode the input data into the quantum circuit. Defaults to qml.AngleEmbedding with X rotation.",
    )
    unary_operator: PauliZ = Field(
        default_factory=PauliZ,
        description="A callable that takes a single argument and returns a transformed version of it. Used to apply unary operations to the quantum circuit outputs before returning them. Defaults to qml.PauliZ measurement.",
    )

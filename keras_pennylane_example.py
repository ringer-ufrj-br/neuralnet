import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import pennylane as qml
import numpy as np
from sklearn.datasets import make_moons

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the number of qubits
n_qubits = 2

# Instantiate a PennyLane device using the default simulator
dev = qml.device("default.qubit", wires=n_qubits)


# Define the quantum circuit
# The interface is set to "torch" so it integrates natively with PyTorch
@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    # AngleEmbedding encodes the inputs into the quantum state
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # BasicEntanglerLayers adds parameterized rotation and entanglement
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    # Return the expectation value of the PauliZ observable on each qubit
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


# Configure the PennyLane TorchLayer
n_layers = 2
weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

from keras.layers import TorchModuleWrapper


def create_hybrid_model():
    """
    Creates a Hybrid Neural Network with a Quantum Circuit as the last representation layer.
    The final output is a single neuron with a sigmoid activation for binary classification.
    """
    inputs = keras.Input(shape=(2,), name="input_features")

    # Classical layers
    x = keras.layers.Dense(8, activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dense(n_qubits, activation="tanh", name="dense_2")(
        x
    )  # scale inputs to the quantum layer

    # Quantum layer: using Keras 3 TorchModuleWrapper
    q_out = TorchModuleWrapper(
        qlayer, output_shape=(None, n_qubits), name="quantum_circuit"
    )(x)

    # Final classical output layer for binary classification
    outputs = keras.layers.Dense(1, activation="sigmoid", name="output_prob")(q_out)

    model = keras.Model(
        inputs=inputs, outputs=outputs, name="Hybrid_Quantum_Classical_Model"
    )
    return model


if __name__ == "__main__":
    # Create the model
    model = create_hybrid_model()
    model.summary()

    # Generate random dummy data for binary classification
    train_samples = 10000
    noise = 0.2
    X, y = make_moons(n_samples=train_samples, noise=noise)

    # Compile the model
    # Note: Keras handles the optimization process, delegating to the PyTorch backend
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    no_fit_metrics = model.evaluate(X, y, verbose=0)
    print(f'\nBefore training - Accuracy: {no_fit_metrics[1]:.4f}, Loss: {no_fit_metrics[0]:.4f}')

    # Train the hybrid model
    print("\nStarting training of the hybrid quantum-classical model...")
    fit_results = model.fit(X, y, epochs=5, batch_size=16)

    print("\nTraining complete!")

    # Evaluate
    after_fit_metrics = model.evaluate(X, y, verbose=0)
    print(f"\nFinal Training Accuracy: {after_fit_metrics[1]:.4f}, Loss: {after_fit_metrics[0]:.4f}")

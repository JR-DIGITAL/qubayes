"""
This script contains code for generative models.
"""
__author__ = "Florian Krebs"
from qiskit.circuit.library import EfficientSU2, RealAmplitudes
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from qubayes.qubayes_tools import counts_to_cpd


class BornMachine(object):

    def __init__(self, n_qubits, n_blocks=0, ansatz_type='RealAmplitudes'):
        # n_blocks is L in the paper
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.params = None
        self.ansatz = None
        self.q_bm = None
        self.ansatz_type = ansatz_type
        self.reset_ansatz()

    def set_params(self, params):
        self.params = params

    def reset_ansatz(self):
        # Reset and create Ansatz with parameters from self.params
        if self.ansatz_type == 'RealAmplitudes':
            self.ansatz = RealAmplitudes(self.n_qubits, reps=self.n_blocks,
                                         entanglement='linear')
        else:
            self.ansatz = EfficientSU2(self.n_qubits, su2_gates=['rz', 'rx'],
                                       reps=self.n_blocks,
                                       entanglement='linear')
        if self.params is None:
            self.params = np.random.normal(0, 0.1, size=self.ansatz.num_parameters)
        param_dict = {param: value for param, value in zip(self.ansatz.parameters, self.params)}
        self.ansatz.assign_parameters(param_dict, inplace=True)

    def print_circuit(self):
        print(self.ansatz.decompose())

    def sample(self, n_samples, return_samples=True):
        # Create a quantum circuit
        circuit = QuantumCircuit(self.n_qubits)
        # Apply a Hadamard gate to each qubit for state preparation
        for qubit in range(self.n_qubits):
            circuit.h(qubit)
        self.reset_ansatz()  # loads self.params
        circuit.compose(self.ansatz, inplace=True)
        circuit.measure_all()

        # Simulate the circuit
        simulator = AerSimulator(method='matrix_product_state')
        compiled_circuit = transpile(circuit, simulator)
        result = simulator.run(compiled_circuit, shots=n_samples, memory=True).result()
        counts = result.get_counts()
        self.q_bm = counts_to_cpd(counts, reverse=True)
        if return_samples:
            samples = result.get_memory()
            out = np.array([[char == '1' for char in string[::-1]] for string in samples], dtype='int32')
            # check
            unique_rows, counts = np.unique(out, axis=0, return_counts=True)
            if len(counts) > 1:
                np.testing.assert_almost_equal(counts[1]/n_samples, self.q_bm[tuple(unique_rows[1])], decimal=3)
        else:
            out = self.q_bm
        return out


class RBM(object):
    def __init__(self, n_visible, n_hidden, seed=None):
        """
        Restricted Boltzmann Machine (binary-binary version).

        Energy:  E(v,h) = -v^T W h - b^T v - c^T h
        Probability: p(v) ∝ ∑_h exp(-E(v,h))

        Args:
            n_visible (int): number of visible units (data dimension)
            n_hidden (int): number of hidden units (latent dimension)
            seed (int): random seed for reproducibility
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # Initialize parameters (small random weights)
        self.W = np.random.normal(0.0, 0.01, size=(n_visible, n_hidden))
        self.b = np.zeros(n_visible)
        self.c = np.zeros(n_hidden)

        # Flattened parameter vector for derivative-free optimizers
        self.params = self._flatten_params()

    # ------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------

    def _flatten_params(self):
        """Flatten parameters into a single vector."""
        return np.concatenate([self.W.flatten(), self.b, self.c])

    def _unflatten_params(self, params):
        """Convert flat parameter vector back to W, b, c."""
        offset = 0
        W_size = self.n_visible * self.n_hidden
        self.W = params[offset : offset + W_size].reshape(self.n_visible, self.n_hidden)
        offset += W_size
        self.b = params[offset : offset + self.n_visible]
        offset += self.n_visible
        self.c = params[offset : offset + self.n_hidden]

    def set_params(self, params):
        """Set model parameters from optimizer-provided vector."""
        self.params = np.array(params, dtype=np.float32)
        self._unflatten_params(self.params)

    # ------------------------------------------------------------
    # Gibbs sampling utilities
    # ------------------------------------------------------------

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sample_bernoulli(self, probs):
        """Sample binary states (0/1) given probabilities."""
        return (np.random.rand(*probs.shape) < probs).astype(np.float32)

    def sample_hidden_given_visible(self, v):
        """Sample h ~ p(h | v) = sigmoid(c + W^T v)."""
        probs_h = self._sigmoid(self.c + np.dot(v, self.W))
        h = self._sample_bernoulli(probs_h)
        return h, probs_h

    def sample_visible_given_hidden(self, h):
        """Sample v ~ p(v | h) = sigmoid(b + W h)."""
        probs_v = self._sigmoid(self.b + np.dot(h, self.W.T))
        v = self._sample_bernoulli(probs_v)
        return v, probs_v

    # ------------------------------------------------------------
    # Main sampling function
    # ------------------------------------------------------------

    def sample(self, n_samples, n_gibbs_steps=10):
        """
        Draw samples v ~ p_theta(v) via block Gibbs sampling.

        Start from random visible states and alternate updates.
        """
        # Initialize visible layer randomly
        v = np.random.binomial(1, 0.5, size=(n_samples, self.n_visible)).astype(np.float32)

        for _ in range(n_gibbs_steps):
            h, _ = self.sample_hidden_given_visible(v)
            v, _ = self.sample_visible_given_hidden(h)

        return v

    # ------------------------------------------------------------
    # Optional: compute unnormalized energy
    # ------------------------------------------------------------

    def energy(self, v):
        """Compute unnormalized energy of visible sample v."""
        # E(v) = -bᵀv - ∑_j log(1 + exp(c_j + (Wᵀv)_j))
        hidden_term = np.sum(np.log(1 + np.exp(self.c + np.dot(v, self.W))), axis=1)
        linear_term = np.dot(v, self.b)
        return -linear_term - hidden_term


class AutoRegressiveModel(object):
    def __init__(self, n_dim, hidden_units=[32, 32], seed=None):
        """
        Args:
            n_dim (int): number of variables (length of sample vector)
            hidden_units (list): hidden layer sizes for the conditional network
            seed (int): random seed for reproducibility
        """
        self.n_dim = n_dim
        self.hidden_units = hidden_units
        self.seed = seed
        self._build_network()
        self.params = self.get_params_vector()  # np.array of shape (n_params,)

    def _build_network(self):
        """Builds a small Keras model that predicts P(x_i=1 | x_{<i})."""
        tf.random.set_seed(self.seed)

        inputs = keras.Input(shape=(self.n_dim,), dtype=tf.float32)
        x = inputs
        for h in self.hidden_units:
            x = layers.Dense(h, activation="relu")(x)
        outputs = layers.Dense(self.n_dim, activation="sigmoid")(x)

        # This network will be used autoregressively
        self.net = keras.Model(inputs, outputs)
        self.param_shapes = [w.shape for w in self.net.trainable_weights]

    def get_params_vector(self):
        """Return all network parameters flattened into a single vector."""
        return tf.concat(
            [tf.reshape(w, [-1]) for w in self.net.get_weights()], axis=0
        ).numpy()

    def set_params(self, params):
        """Sets model parameters from a flat vector."""
        self.params = np.array(params, dtype=np.float32)
        new_weights = []
        offset = 0
        for shape in self.param_shapes:
            size = np.prod(shape)
            new_weights.append(
                np.reshape(self.params[offset:offset + size], shape)
            )
            offset += size
        self.net.set_weights(new_weights)

    def sample(self, n_samples):
        """
        Sample n_samples binary vectors x ~ p_theta(x)
        using the autoregressive factorization.
        """
        samples = np.zeros((n_samples, self.n_dim), dtype=np.float32)
        for i in range(self.n_dim):
            # condition on previously sampled bits
            logits = self.net(samples).numpy()  # shape: (n_samples, n_dim)
            probs = logits[:, i]
            # sample x_i ~ Bernoulli(p_i)
            samples[:, i] = np.random.binomial(1, probs)  # one sample for each prob
        return samples

def test_ar():
    n_dim = 10
    model = AutoRegressiveModel(n_dim, hidden_units=[16])

    # Random initial parameters (for COBYLA)
    theta = model.get_params_vector()

    # Evaluate samples for a given parameter vector
    model.set_params(theta)
    x_samples = model.sample(n_samples=100)
    print("Samples shape:", x_samples.shape)
    print("First few samples:\n", x_samples[:5])


def test_rbm():
    rbm = RBM(n_visible=8, n_hidden=4, seed=42)

    # Get flat parameter vector (for COBYLA or other optimizer)
    theta = rbm.params

    # Define your black-box loss function
    def loss_function(samples):
        # Example loss: encourage low Hamming weight
        return np.mean(np.sum(samples, axis=1))

    # Objective for optimizer
    def objective(params):
        rbm.set_params(params)
        samples = rbm.sample(n_samples=200, n_gibbs_steps=20)
        return loss_function(samples)

    # Example evaluation (without running an optimizer)
    val = objective(theta)
    print("Initial objective value:", val)
    print("Example samples:\n", rbm.sample(5))


if __name__ == "__main__":
    # bm = BornMachine(1)
    test_ar()
    test_rbm()



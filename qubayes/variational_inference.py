"""
This script contains the code for implementing variational inference based
on the paper "Variational inference with a quantum computer" by Marcello
Benedetti et al., 2021.
"""
__author__ = "Florian Krebs"
from sprinkler_example import create_graph as create_sprinkler_graph
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from copy import deepcopy



class Optimizer(object):

    def __init__(self, born_machine, bayes_net, classifier=None, n_iterations=100, learning_rate=0.003):
        self.born_machine = born_machine
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.bayes_net = bayes_net
        self.classifier = classifier
        self.initialize_classifier()

    def estimate_gradient(self):
        # Use parameter shift rule for estimation, as outlined in B15 in the paper
        shift = np.pi / 2
        gradients = np.zeros(self.born_machine.params.shape)

        for i in range(len(self.born_machine.params)):
            # Original parameters
            bm = {'plus': deepcopy(self.born_machine),
                  'minus': deepcopy(self.born_machine)}
            # apply shifts
            bm['plus'].params[i] += shift
            bm['minus'].params[i] -= shift
            md = dict()

            for key in ['plus', 'minus']:
                # Sample 50 points
                samples_crs = bm[key].sample(50)
                # classify 50 points
                # TODO: Really logit?
                logit_d = self.classifier.predict(samples_crs)[:, 0]
                # compute P(x|z) for the 50 points
                loglik = self.bayes_net.compute_log_likelihood(samples_crs)
                # compute the mean difference (logit(d_i) - log(p(x_i|z_i))) ->
                md[key] = (logit_d - loglik).mean()

            # compute the gradient as (md_plus - md_minus) / 2
            gradients[i] = (md['plus'] - md['minus']) / 2
        return gradients

    def initialize_classifier(self):
        if self.classifier is None:
            self.classifier = tf.keras.Sequential([
                layers.Input(shape=(self.born_machine.n_qubits,)),
                layers.Dense(6, activation='relu'),
                layers.Dense(1)
            ])

    def train_classifier(self, train_x, train_y, learning_rate=0.03):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.classifier.compile(optimizer=optimizer,
                                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # we already have a sigmoid after the last layer
                                metrics=['accuracy'])
        self.classifier.fit(x=train_x, y=train_y, epochs=20, batch_size=10)
        return

    def compute_loss(self):
        return 1

    def optimize(self):
        s_prior = self.bayes_net.sample_from_joint(100)
        for i in range(self.n_iterations):
            # Draw 100 sample from the born machine
            s_bm = self.born_machine.sample(100)

            # Train Classifier with {S_prior + S_born} to distinguish prior and born machine
            x_train = np.vstack((s_bm, s_prior))
            y_train = np.zeros((s_prior.shape[0] + s_bm.shape[0],))  # 0 ... born machine, 1 ... prior
            y_train[s_bm.shape[0]:] = 1
            self.train_classifier(x_train, y_train)

            # Calculate the gradient using the parameter-shift rule
            gradient = self.estimate_gradient()

            # Update the parameter (for maximization, we add the gradient)
            self.bm.params += self.learning_rate * gradient

            # Evaluate the function at the new theta
            loss = self.compute_loss()

            # Print the current theta and function value
            print(f"Iteration {i + 1}: loss = {loss:.4f}, mean gradient = {gradient.mean():.4f}")
        return self.bm


class BornMachine(object):

    def __init__(self, n_qubits, n_blocks=0):
        # n_blocks is L in the paper
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.params = None
        self.ansatz = EfficientSU2(n_qubits,
                                   su2_gates=['rz', 'rx'],
                                   reps=n_blocks,
                                   entanglement='linear')
        self.initialize()

    def initialize(self, std=0.01):
        self.params = np.random.normal(0, std, size=self.ansatz.num_parameters)

    def print_circuit(self):
        print(self.ansatz.decompose())

    def sample(self, n_samples):
        param_dict = {param: value for param, value in zip(self.ansatz.parameters, self.params)}
        self.ansatz.assign_parameters(param_dict, inplace=True)

        # Create a quantum circuit
        circuit = QuantumCircuit(self.n_qubits)
        # Apply a Hadamard gate to each qubit for state preparation
        for qubit in range(self.n_qubits):
            circuit.h(qubit)
        circuit.compose(self.ansatz, inplace=True)
        circuit.measure_all()

        # Simulate the circuit
        simulator = AerSimulator(method='matrix_product_state')
        # simulator = Aer.get_backend('aer_simulator')
        compiled_circuit = transpile(circuit, simulator)
        result = simulator.run(compiled_circuit, shots=n_samples, memory=True).result()
        # counts = dict(result.get_counts())
        samples = result.get_memory()
        samples = np.array([[char == '1' for char in string] for string in samples], dtype='int32')
        return samples


class SprinklerBN(object):

    def __init__(self):
        self.graph = create_sprinkler_graph()

    def sample_from_joint(self, n_samples):
        s_prior = self.graph.sample_from_graph(n_samples)
        s_prior_crs = s_prior[0][:3, :][[0, 2, 1], :].transpose()  # convert to C, R, S: 100 x 3
        return s_prior_crs

    def compute_log_p_w_crs(self, samples_crs):
        # Compute the log likelihood P(W=1 | C, R, S) = P(W=1 | R, S)
        log_lik = np.zeros((samples_crs.shape[0],))
        for i in range(samples_crs.shape[0]):
            c, r, s = samples_crs[i, :]
            # TODO: check for small probabilities
            log_lik[i] = np.log(max([1e-2, self.graph.nodes['wet'].data[1, s, r]]))
        return log_lik

    def compute_log_likelihood(self, n_samples):
        return self.compute_log_p_w_crs(n_samples)





if __name__ == "__main__":
    # Create BN object
    sprinkler_bn = SprinklerBN()
    # Initialize a born machine
    bm = BornMachine(3, n_blocks=1)
    bm.print_circuit()
    # Optimize it
    opt = Optimizer(bm, sprinkler_bn, n_iterations=3, learning_rate=0.003)
    opt.optimize()

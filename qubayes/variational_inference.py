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

    def estimate_gradient(self, n_samples=100):
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
                samples_crs = bm[key].sample(n_samples)
                # classify 50 points
                # TODO: Really logit?
                logit_d = self.classifier.predict(samples_crs, verbose=0)[:, 0]
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
        self.classifier.fit(x=train_x, y=train_y, epochs=50, batch_size=10,
                            verbose=0)
        return

    def compute_loss(self, samples_crs):
        logit_d = self.classifier.predict(samples_crs, verbose=0)[:, 0]
        loglik = self.bayes_net.compute_log_likelihood(samples_crs)
        return (logit_d - loglik).mean()

    def optimize(self):
        s_prior = self.bayes_net.sample_from_joint(100)
        tvd = np.zeros((self.n_iterations,))
        for i in range(self.n_iterations):
            # Draw 100 sample from the born machine
            s_bm = self.born_machine.sample(100)

            # Train Classifier with {S_prior + S_born} to distinguish prior and born machine
            x_train = np.vstack((s_bm, s_prior))
            y_train = np.zeros((s_prior.shape[0] + s_bm.shape[0],))  # 0 ... born machine, 1 ... prior
            y_train[s_bm.shape[0]:] = 1
            self.train_classifier(x_train, y_train)

            # Calculate the gradient using the parameter-shift rule
            gradients = self.estimate_gradient()

            # Update the parameter (for maximization, we add the gradient)
            self.born_machine.params += self.learning_rate * gradients

            # Evaluate the function at the new theta
            loss = self.compute_loss(s_bm)
            tvd[i] = self.compute_tvd(s_bm)

            # Print the current theta and function value
            print(f"Iteration {i + 1}: tvd = {tvd[i]:.4f}, loss = {loss:.4f}, mean gradient = {gradients.mean():.4f}")
        return self.born_machine, tvd

    def compute_tvd(self, samples):
        # Compute total variation distance (The largest absolute difference
        # between the probabilities that the two probability distributions
        # assign to the same event.).
        return self.bayes_net.compute_tvd(samples)


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
        compiled_circuit = transpile(circuit, simulator)
        result = simulator.run(compiled_circuit, shots=n_samples, memory=True).result()
        samples = result.get_memory()
        samples = np.array([[char == '1' for char in string] for string in samples], dtype='int32')
        return samples


class SprinklerBN(object):

    def __init__(self, random_cpd=True):
        self.random_cpd = random_cpd
        self.graph = create_sprinkler_graph()
        if random_cpd:
            self.graph.nodes['cloudy'].data[0] = np.random.uniform(low=0.01, high=0.99)
            self.graph.nodes['cloudy'].data[1] = 1. - self.graph.nodes['cloudy'].data[0]
            self.graph.nodes['rain'].data[0, 0] = np.random.uniform(low=0.01, high=0.99)
            self.graph.nodes['rain'].data[1, 0] = 1. - self.graph.nodes['rain'].data[0, 0]
            self.graph.nodes['rain'].data[0, 1] = np.random.uniform(low=0.01, high=0.99)
            self.graph.nodes['rain'].data[1, 1] = 1. - self.graph.nodes['rain'].data[0, 1]
            self.graph.nodes['sprinkler'].data[0, 0] = np.random.uniform(low=0.01, high=0.99)
            self.graph.nodes['sprinkler'].data[1, 0] = 1. - self.graph.nodes['sprinkler'].data[0, 0]
            self.graph.nodes['sprinkler'].data[0, 1] = np.random.uniform(low=0.01, high=0.99)
            self.graph.nodes['sprinkler'].data[1, 1] = 1. - self.graph.nodes['sprinkler'].data[0, 1]
            self.graph.nodes['wet'].data[0, 0, 0] = np.random.uniform(low=0.01, high=0.99)
            self.graph.nodes['wet'].data[1, 0, 0] = 1. - self.graph.nodes['wet'].data[0, 0, 0]
            self.graph.nodes['wet'].data[0, 0, 1] = np.random.uniform(low=0.01, high=0.99)
            self.graph.nodes['wet'].data[1, 0, 1] = 1. - self.graph.nodes['wet'].data[0, 0, 1]
            self.graph.nodes['wet'].data[0, 1, 0] = np.random.uniform(low=0.01, high=0.99)
            self.graph.nodes['wet'].data[1, 1, 0] = 1. - self.graph.nodes['wet'].data[0, 1, 0]
            self.graph.nodes['wet'].data[0, 1, 1] = np.random.uniform(low=0.01, high=0.99)
            self.graph.nodes['wet'].data[1, 1, 1] = 1. - self.graph.nodes['wet'].data[0, 1, 1]

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

    def compute_tvd(self, samples_crs):
        # Total variation distance between Q and P(C, R, S | W = 1)
        # 1) Get the counts of the samples -> Q
        unique_rows, unique_counts = np.unique(samples_crs, axis=0, return_counts=True)
        estimation = np.zeros((2, 2, 2), dtype=float)  # C, R, S
        for c in range(2):
            for r in range(2):
                for s in range(2):
                    idx = (unique_rows == np.array([c, r, s])).all(axis=1)
                    if idx.any():
                        estimation[c, r, s] = float(unique_counts[idx][0]) / samples_crs.shape[0]

        # 2) Get the exact probabilities -> P
        posterior = np.zeros((2, 2, 2))  # C, R, S
        for c in range(2):
            prob = self.graph.nodes['cloudy'].data[c]
            for r in range(2):
                prob2 = prob * self.graph.nodes['rain'].data[r, c]
                for s in range(2):
                    prob3 = prob2 * self.graph.nodes['sprinkler'].data[s, c]
                    prob3 *= self.graph.nodes['wet'].data[1, s, r]
                    posterior[c, r, s] = prob3
        posterior /= posterior.sum()

        # 3) Find max distance
        tvd = (abs(posterior - estimation)).max()
        return tvd


if __name__ == "__main__":
    # Create BN object
    sprinkler_bn = SprinklerBN()
    # Initialize a born machine
    bm = BornMachine(3, n_blocks=1)
    bm.print_circuit()
    # Optimize it
    opt = Optimizer(bm, sprinkler_bn, n_iterations=200, learning_rate=0.003)
    bm_opt = opt.optimize()

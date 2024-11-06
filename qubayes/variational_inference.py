"""
This script contains the code for implementing variational inference based
on the paper "Variational inference with a quantum computer" by Marcello
Benedetti et al., 2021.
"""
__author__ = "Florian Krebs"
from sprinkler_example import create_graph as create_sprinkler_graph
from qiskit.circuit.library import EfficientSU2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from copy import deepcopy
import matplotlib.pyplot as plt


class OptimalClassifier(object):

    def __init__(self, bayes_net):
        self.q_crs = None
        self.p_crs = bayes_net.compute_p_crs()

    def train(self, train_x, train_y, learning_rate=None):
        # learn q(C, R, S | W = 1) from train_x
        train_x = train_x[train_y == 0, :]  # get only samples from born machine
        unique_rows, unique_counts = np.unique(train_x, axis=0, return_counts=True)
        estimation = np.zeros((2, 2, 2), dtype=float)  # C, R, S
        for c in range(2):
            for r in range(2):
                for s in range(2):
                    idx = (unique_rows == np.array([c, r, s])).all(axis=1)
                    if idx.any():
                        estimation[c, r, s] = float(unique_counts[idx][0]) / train_x.shape[0]
        self.q_crs = estimation

    def predict(self, samples):
        self.train(samples, np.zeros((samples.shape[0],)))  # update q_crs
        pred = np.zeros((samples.shape[0],))
        for i in range(samples.shape[0]):
            (c, r, s) = samples[i, :]
            pred[i] = self.q_crs[c, r, s] / (self.q_crs[c, r, s] + self.p_crs[c, r, s])
        return pred


class Classifier(object):

    def __init__(self, n_inputs):
        self.model = tf.keras.Sequential([
            layers.Input(shape=(n_inputs,)),
            layers.Dense(6, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

    def train(self, train_x, train_y, learning_rate=0.03):
        # shuffle datasets
        idx = np.random.permutation(train_x.shape[0])
        train_x = train_x[idx, :]
        train_y = train_y[idx]
        split = 0.2
        idx = int(train_x.shape[0] * split)
        val_x = train_x[:idx, :]
        val_y = train_y[:idx]
        train_x = train_x[idx:, :]
        train_y = train_y[idx:]

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Define the early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',         # Monitor validation loss
            patience=20,                # Wait for 5 epochs of no improvement
            restore_best_weights=True   # Restore the model to the best epoch
        )
        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  # we already have a sigmoid after the last layer
                           metrics=['accuracy'])
        history = self.model.fit(x=train_x, y=train_y,
                                 validation_data=(val_x, val_y),
                                 epochs=200, batch_size=10,
                                 verbose=0, callbacks=[early_stopping])

        # Calculate the best epoch index
        best_epoch = early_stopping.stopped_epoch - early_stopping.patience + 1

        # Retrieve the validation accuracy of the best epoch
        best_val_accuracy = history.history['val_accuracy'][best_epoch]
        print(f"- Validation accuracy {best_val_accuracy:.2f} at best epoch {best_epoch}.")

        return history

    def predict(self, samples):
        # if prob > 0.5 => class = 1
        return self.model.predict(samples, verbose=0)[:, 0]  # return 1d array


class Optimizer(object):

    def __init__(self, born_machine, bayes_net, n_iterations=100, learning_rate=0.003, use_optimal_clf=False):
        self.born_machine = born_machine
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.bayes_net = bayes_net
        if use_optimal_clf:
            self.classifier = OptimalClassifier(bayes_net)
        else:
            self.classifier = Classifier(n_inputs=self.born_machine.n_qubits)

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
                p_prior = self.classifier.predict(samples_crs)
                p_bm = 1. - p_prior
                logit_d = np.log(p_bm / (1 - p_bm))
                # compute P(x|z) for the 50 points
                loglik = self.bayes_net.compute_log_likelihood(samples_crs)
                # compute the mean difference (logit(d_i) - log(p(x_i|z_i))) ->
                md[key] = (logit_d - loglik).mean()

            # compute the gradient as (md_plus - md_minus) / 2
            gradients[i] = (md['plus'] - md['minus']) / 2
        return gradients

    def compute_loss(self, samples_crs=None):
        if samples_crs is None:
            samples_crs = self.born_machine.sample(100)
        p_prior = self.classifier.predict(samples_crs)
        p_born = 1. - p_prior
        logit_d = np.log(p_born / (1 - p_born))
        loglik = self.bayes_net.compute_log_likelihood(samples_crs)
        return (logit_d - loglik).mean()

    def optimize(self):
        metrics = {'tvd': np.zeros((self.n_iterations,)),
                   'loss': np.zeros((self.n_iterations,))}
        for i in range(self.n_iterations):
            # Draw 100 sample from the born machine
            s_bm = self.born_machine.sample(100)
            s_prior = self.bayes_net.sample_from_joint(100)

            # Train Classifier with {S_prior + S_born} to distinguish prior and born machine
            x_train = np.vstack((s_bm, s_prior))
            y_train = np.zeros((s_prior.shape[0] + s_bm.shape[0],))  # 0 ... born machine, 1 ... prior
            y_train[s_bm.shape[0]:] = 1
            history = self.classifier.train(x_train, y_train)

            # plt.figure()
            # plt.plot(history.history['loss'])
            # plt.plot(history.history['val_loss'])
            # plt.legend(['train_loss', 'val_loss'])
            # plt.savefig(fr'C:\Users\krf\projects\Quanco\git\qubayes\figs\vi_classifier\learning_curve_it{i}.png')

            # Calculate the gradient using the parameter-shift rule
            gradients = self.estimate_gradient()

            # Update the parameter (for maximization, we add the gradient)
            self.born_machine.params += self.learning_rate * gradients

            # Evaluate the function at the new theta
            metrics['loss'][i] = self.compute_loss()
            metrics['tvd'][i] = self.compute_tvd(s_bm)

            # Print the current theta and function value
            print(f"Iteration {i + 1}: tvd = {metrics['tvd'][i]:.4f}, loss = {metrics['loss'][i]:.4f},"
                  f" mean gradient = {gradients.mean():.4f}")
        return self.born_machine, metrics

    def compute_tvd(self, samples):
        # Compute total variation distance (The largest absolute difference
        # between the probabilities that the two probability distributions
        # assign to the same event.).
        return self.bayes_net.compute_tvd(samples)


# class DFOptimizer(Optimizer):
#
#     def __init__(self):


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
            log_lik[i] = np.log(max([1e-3, self.graph.nodes['wet'].data[1, s, r]]))
        return log_lik

    def compute_log_likelihood(self, n_samples):
        return self.compute_log_p_w_crs(n_samples)

    def compute_p_crs(self):
        p_crs = np.zeros((2, 2, 2))  # C, R, S
        for c in range(2):
            prob = self.graph.nodes['cloudy'].data[c]
            for r in range(2):
                prob2 = prob * self.graph.nodes['rain'].data[r, c]
                for s in range(2):
                    p_crs[c, r, s] = prob2 * self.graph.nodes['sprinkler'].data[s, c]
        return p_crs / p_crs.sum()

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


def plot_optimization_metrics(metrics, save=False):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(metrics['loss'], label='Loss according to Eq. 7')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss (Eq. 7)')
    ax[0].legend()
    ax[1].plot(metrics['tvd'], label='TVD between q(z|x) and p(z|x)')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('TVD')
    ax[1].legend()
    if save:
        plt.savefig(fr'C:\Users\krf\projects\Quanco\git\qubayes\figs\vi_classifier\optimization.png')
    else:
        plt.show()


if __name__ == "__main__":
    # Create BN object
    sprinkler_bn = SprinklerBN()
    # Initialize a born machine
    bm = BornMachine(3, n_blocks=0)
    bm.print_circuit()
    # Optimize it
    opt = Optimizer(bm, sprinkler_bn,
                    n_iterations=400,
                    learning_rate=0.003,
                    use_optimal_clf=True)
    bm_opt, metrics = opt.optimize()
    plot_optimization_metrics(metrics, save=1)



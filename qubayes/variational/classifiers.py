"""
This script contains code for classifiers.
"""
__author__ = "Florian Krebs"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import itertools


class OptimalClassifier(object):

    def __init__(self, bayes_net):
        self.bn = bayes_net
        self.q_posterior = None
        self.p_prior = bayes_net.compute_p_prior()

    def train(self, train_x, train_y, learning_rate=None):
        # learn q(C, R, S | W = 1) from train_x
        train_x = train_x[train_y == 0, :]  # get only samples from born machine
        unique_rows, unique_counts = np.unique(train_x, axis=0, return_counts=True)
        # TODO: this works only for a single evidence variable
        lst = list(itertools.product([0, 1], repeat=self.bn.graph.n_variables-1))
        estimation = np.zeros((2,) * (self.bn.graph.n_variables-1), dtype=float)
        for c in lst:
            idx = (unique_rows == np.array([c])).all(axis=1)
            if idx.any():
                estimation[c] = float(unique_counts[idx][0]) / train_x.shape[0]
        self.q_posterior = estimation
        return None

    def predict(self, samples, labels=None):
        # According to Eq. 5, predict p_prior
        if labels is None:
            labels = np.zeros((samples.shape[0],))
        self.train(samples, labels)  # update q_crs using samples from bm (class 0)
        pred = np.zeros((samples.shape[0],))
        for i in range(samples.shape[0]):
            idx = tuple(samples[i, :])
            pred[i] = self.q_posterior[idx] / (self.q_posterior[idx] + self.p_prior[idx])
        return pred

    def compute_loss(self, train_x, train_y):
        # According to Eq. 4
        # 0 ... born machine, 1 ... prior
        p_prior = self.predict(train_x, labels=train_y)
        # born machine samples
        E_log_bm = (np.log(1 - p_prior[train_y == 0])).mean()
        # prior
        E_log_prior = (np.log(p_prior[train_y == 1])).mean()
        return E_log_bm + E_log_prior


class MLP_Classifier(object):

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

    def compute_loss(self, train_x, train_y):
        # According to Eq. 4
        # 0 ... born machine, 1 ... prior
        p_prior = self.predict(train_x)
        # born machine samples
        E_log_bm = (np.log(1 - p_prior[train_y == 0])).mean()
        # prior
        E_log_prior = (np.log(p_prior[train_y == 1])).mean()
        return E_log_bm + E_log_prior

    def predict(self, samples):
        # if prob > 0.5 => class = 1 (prior)
        return self.model.predict(samples, verbose=0)[:, 0]  # return 1d array



if __name__ == "__main__":
    main()

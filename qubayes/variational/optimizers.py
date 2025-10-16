"""
This script contains code for optimizers
"""
__author__ = "Florian Krebs"

import numpy as np
from copy import deepcopy
from qiskit_algorithms import optimizers


def logit(x):
    eps = 1e-5
    if x > (1 - eps):
        x = 1 - eps
    elif x < eps:
        x = eps
    return np.log(x / (1 - x))


class Optimizer(object):

    def __init__(self, born_machine, bayes_net, classifier, n_iterations=100, learning_rate=0.003):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.bayes_net = bayes_net
        self.classifier = classifier
        self.born_machine = born_machine

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
                # compute P(x|z) for the 50 points
                loglik = self.bayes_net.compute_log_likelihood(samples_crs)
                # compute the mean difference (logit(d_i) - log(p(x_i|z_i))) ->
                md[key] = (logit(p_bm) - loglik).mean()

            # compute the gradient as (md_plus - md_minus) / 2
            gradients[i] = (md['plus'] - md['minus']) / 2
        return gradients

    def compute_kl_loss(self, samples_crs=None):
        # Compute L_KL loss as defined in Eq. 7.
        if samples_crs is None:
            samples_crs = self.born_machine.sample(100)
        p_prior = self.classifier.predict(samples_crs)
        p_born = 1. - p_prior
        loglik = self.bayes_net.compute_log_likelihood(samples_crs)
        return (logit(p_born) - loglik).mean()

    def optimize(self):
        metrics = {'tvd': np.zeros((self.n_iterations,)),
                   'kl_loss': np.zeros((self.n_iterations,)),
                   'ce_loss': np.zeros((self.n_iterations,))}
        for i in range(self.n_iterations):
            # Draw 100 sample from the born machine
            s_bm = self.born_machine.sample(100)
            s_prior = self.bayes_net.sample_from_prior(100)

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
            metrics['kl_loss'][i] = self.compute_kl_loss(s_bm)
            metrics['tvd'][i] = self.compute_tvd(s_bm)
            metrics['ce_loss'][i] = self.classifier.compute_loss(x_train, y_train)

            # Print the current theta and function value
            print(f"Iteration {i + 1}: tvd = {metrics['tvd'][i]:.4f}, born loss = {metrics['kl_loss'][i]:.4f},"
                  f" clf loss = {metrics['ce_loss'][i]:.4f}")
        return self.born_machine, metrics

    def compute_tvd(self, samples):
        # Compute total variation distance (The largest absolute difference
        # between the probabilities that the two probability distributions
        # assign to the same event.).
        return self.bayes_net.compute_tvd(samples)


class DerivativeFreeOptimizer(object):

    def __init__(self, born_machine, bayes_net, classifier, n_iterations=100,
                 learning_rate=0.003, method='COBYLA'):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.bayes_net = bayes_net
        self.classifier = classifier
        self.born_machine = born_machine
        self.method = method
        self.params = None

    def compute_kl_loss(self, theta_values):
        # An objective function takes parameters as input and outputs the loss.
        n_samples = 1000
        self.born_machine.set_params(theta_values)
        samples = self.born_machine.sample(n_samples)
        q_bm = self.born_machine.q_bm
        p_prior = self.bayes_net.compute_p_prior()
        logliks = self.bayes_net.compute_log_likelihood(samples)
        loss = 0
        for i in range(samples.shape[0]):
            s = samples[i, :]
            # If q and P(z|x) match, the loss should be np.log(self.bayes_net.compute_p_wet(wet=1))
            d_bm = q_bm[tuple(s)] / (q_bm[tuple(s)] + p_prior[tuple(s)])  # optimal classifier
            loss += (logit(d_bm) - logliks[i])
            # loss += np.log(q_bm[tuple(s)] / p_prior[tuple(s)]) - logliks[i]
        return loss / samples.shape[0]

    def optimize(self):

        optimizer = getattr(optimizers, self.method)
        info = {'parameters': [], 'loss': []}

        if self.method == 'COBYLA':
            def callback(parameters):  # storing intermediate info
                info['parameters'].append(parameters)
        elif self.method == 'GradientDescent':
            def callback(nfev, parameters, loss_value, gradient_norm):  # storing intermediate info
                info['parameters'].append(parameters)
                info['loss'].append(loss_value)
        opt = optimizer(maxiter=self.n_iterations, callback=callback, tol=1e-10)
        opt_results = opt.minimize(self.compute_kl_loss, self.born_machine.params)
        n_iterations = len(info['parameters'])
        metrics = {'tvd': np.zeros(n_iterations),
                   'kl_loss': np.zeros(n_iterations),
                   'ce_loss': np.zeros(n_iterations)}
        # posterior = self.bayes_net.compute_posterior()
        for i in range(len(info['parameters'])):
            # Print progress
            metrics['kl_loss'][i] = self.compute_kl_loss(info['parameters'][i])
            self.born_machine.params = info['parameters'][i]
            pred = self.born_machine.sample(1000, return_samples=False)
            metrics['tvd'][i] = self.compute_tvd()
            posterior = self.bayes_net.compute_posterior()
            print(f"Iteration {i + 1}: Loss = {metrics['kl_loss'][i]:.4f}, "
                  f"Pred = {pred.flat[0]:.3f}, "
                  f"True = {posterior.flat[0]:.3f}")

        return self.born_machine, metrics

    def compute_tvd(self):
        posterior = self.bayes_net.compute_posterior()
        tvd = (abs(posterior - self.born_machine.q_bm)).max()
        return tvd



if __name__ == "__main__":
    main()

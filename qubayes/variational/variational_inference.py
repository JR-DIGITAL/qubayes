"""
This script contains the code for implementing variational inference based
on the paper "Variational inference with a quantum computer" by Marcello
Benedetti et al., 2021.
"""
__author__ = "Florian Krebs"

from qubayes.sprinkler_example import SprinklerBN
import numpy as np
import matplotlib.pyplot as plt
from qubayes.qubayes_tools import Node, Graph
from qubayes.qubayes_tools import BayesNet
from optimizers import DerivativeFreeOptimizer
from classifiers import OptimalClassifier
from generative_models import BornMachine, RBM, AutoRegressiveModel


class SimpleBN(BayesNet):

    def __init__(self):
        rain = Node('rain', data=np.array([0.5, 0.5]))
        wet = Node('wet', data=np.array([[0.8, 0.1],
                                         [0.2, 0.9]]),
                   parents=['rain'])
        graph = Graph({'rain': rain, 'wet': wet})
        super().__init__(graph)
        return

    def compute_p_prior(self):
        return self.graph.marginalize_all_but(['rain'])

    def compute_log_likelihood(self, samples, wet=1):
        # Compute the log likelihood P(W=1 | R)
        log_lik = np.zeros((samples.shape[0],))
        for i in range(samples.shape[0]):
            r = samples[i, :]
            log_lik[i] = np.log(max([1e-3, self.graph.nodes['wet'].data[wet, r][0]]))
        return log_lik


class SimpleBN2(BayesNet):

    def __init__(self):
        rain = Node('rain', data=np.array([0.3, 0.7]))
        sprinkler = Node('sprinkler', data=np.array([0.8, 0.2]))
        wet = Node('wet', data=np.array([[[0.9, 0.3], [0.2, 0.1]],
                                         [[0.1, 0.7], [0.8, 0.9]]]),
                   parents=['rain', 'sprinkler'])
        graph = Graph({'rain': rain, 'sprinkler': sprinkler, 'wet': wet})
        super().__init__(graph)
        return

    def compute_p_wet(self, wet=1):
        p = self.graph.marginalize_all_but(['wet'])
        return p[wet]

    def compute_p_prior(self):
        # Compute the prior P(R, S)
        return self.graph.marginalize_all_but(['rain', 'sprinkler'])

    def compute_log_likelihood(self, samples, wet=1):
        # Compute the log likelihood P(W=1 | R, S)
        log_lik = np.zeros((samples.shape[0],))
        for i in range(samples.shape[0]):
            (r, s) = samples[i, :]
            log_lik[i] = np.log(max([1e-3, self.graph.nodes['wet'].data[wet, r, s]]))
        return log_lik


def plot_optimization_metrics(metrics, save_fln=None):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(metrics['kl_loss'], label='Loss according to Eq. 7')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss (Eq. 7)')
    ax[0].legend()
    ax[1].plot(metrics['tvd'], label='TVD between q(z|x) and p(z|x)')
    if metrics['ce_loss'].max() > metrics['ce_loss'].min():
        ax[1].plot(metrics['ce_loss'], label='Classifier loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('TVD')
    ax[1].legend()
    if save_fln is None:
        plt.show()
    else:
        plt.savefig(save_fln)
        print(f'Saved figure to {save_fln}')


def main():

    # Create BN object
    bn = SprinklerBN(random_cpd=True)
    # bn = SimpleBN2()

    # Initialize a born machine
    gen_model = BornMachine(len(bn.graph.nodes)-1, n_blocks=1)
    # gen_model = RBM(n_visible=8, n_hidden=4, seed=42)
    # bm = OptimalBornMachine(bn)
    # bm.print_circuit()

    # Classifier
    classifier = OptimalClassifier(bn)
    # classifier = MLP_Classifier(n_inputs=bm.n_qubits)

    # Optimize it
    # opt = Optimizer(bm, bn, classifier, n_iterations=500, learning_rate=0.003)
    opt = DerivativeFreeOptimizer(gen_model, bn, classifier, n_iterations=400, learning_rate=0.003)
    bm_opt, metrics = opt.optimize()
    plot_optimization_metrics(metrics)
    return


if __name__ == "__main__":
    main()



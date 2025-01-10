"""
This script implements the sprinkler Bayesian network network from Kevin
Murphys lecture notes: https://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html
"""
__author__ = "Florian Krebs"

import numpy as np
from qubayes.qubayes_tools import QBNQuery, Node, Graph, BayesNet


class SprinklerBN(BayesNet):

    def __init__(self, random_cpd=True):
        self.random_cpd = random_cpd
        self.graph = self.create_graph()
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

    @staticmethod
    def create_graph():
        cloudy = Node('cloudy', data=np.array([0.5, 0.5]))
        sprinkler = Node('sprinkler', data=np.array([[0.5, 0.9],
                                                     [0.5, 0.1]]),
                         parents=['cloudy'])
        rain = Node('rain', data=np.array([[0.8, 0.2],
                                           [0.2, 0.8]]),
                    parents=['cloudy'])
        probs_wet = np.array([[[1.0, 0.1],  # shape (wet, sprinkler, rain)
                               [0.1, 0.01]],
                              [[0.0, 0.9],
                               [0.9, 0.99]]])
        wet = Node('wet', data=probs_wet,
                   parents=['sprinkler', 'rain'])
        bn = Graph({'cloudy': cloudy, 'sprinkler': sprinkler, 'rain': rain, 'wet': wet})
        return bn

    def compute_log_likelihood(self, samples, wet=1):
        # Compute the log likelihood P(W=1 | C, R, S) = P(W=1 | R, S)
        log_lik = np.zeros((samples.shape[0],))
        for i in range(samples.shape[0]):
            c, r, s = samples[i, :]
            log_lik[i] = np.log(max([1e-3, self.graph.nodes['wet'].data[wet, s, r]]))
        return log_lik

    def compute_p_prior(self):
        p_crs = np.zeros((2, 2, 2))  # C, R, S
        for c in range(2):
            prob = self.graph.nodes['cloudy'].data[c]
            for r in range(2):
                prob2 = prob * self.graph.nodes['rain'].data[r, c]
                for s in range(2):
                    p_crs[c, r, s] = prob2 * self.graph.nodes['sprinkler'].data[s, c]
        return p_crs / p_crs.sum()

    def compute_posterior(self, wet=1):
        # Get the exact probabilities -> P(C, R, S | W = 1)
        posterior = np.zeros((2, 2, 2))  # C, R, S
        for c in range(2):
            prob = self.graph.nodes['cloudy'].data[c]
            for r in range(2):
                prob2 = prob * self.graph.nodes['rain'].data[r, c]
                for s in range(2):
                    prob3 = prob2 * self.graph.nodes['sprinkler'].data[s, c]
                    prob3 *= self.graph.nodes['wet'].data[wet, s, r]
                    posterior[c, r, s] = prob3
        posterior /= posterior.sum()
        return posterior


class QuerySprinkler(QBNQuery):

    def __init__(self):
        super(QuerySprinkler, self).__init__()
        self.graph_orig = SprinklerBN.create_graph()
        self.rebuild_qbn()


def main():
    QS = QuerySprinkler()
    QS.target = {'sprinkler': 'sprinkler1'}
    QS.evidence = {'wet': 'wet1'}
    n_shots = 8000

    # Sample from QBN and compute P(S=1 | W=1) and P(R=1 | W=1) manually
    result = QS.qbn.perform_sampling(shots=n_shots)
    P_W1 = 0
    for k, v in result.items():
        if k[0] == '1':
            P_W1 += v
    P_W1 = P_W1 / n_shots

    # Query 1: P(R=1|W=1)
    P_R1W1 = 0
    for k, v in result.items():
        if k[0] == '1' and k[1] == '1':
            P_R1W1 += v
    P_R1W1 = P_R1W1 / n_shots
    QS.target = {'rain': 'rain1'}
    QS.evidence = {'wet': 'wet1'}
    exact_R1_W1 = QS.get_true_result()
    print(f'P(R=1|W=1)={P_R1W1 / P_W1:.3f}, true={exact_R1_W1:.3f}')  # True: 0.708

    # Query 2: P(S=1|W=1)
    P_S1W1 = 0
    for k, v in result.items():
        if k[0] == '1' and k[2] == '1':
            P_S1W1 += v
    P_S1W1 = P_S1W1 / n_shots
    QS.target = {'sprinkler': 'sprinkler1'}
    QS.evidence = {'wet': 'wet1'}
    exact_S1_W1 = QS.get_true_result()
    print(f'P(S=1|W=1)={P_S1W1 / P_W1:.3f}, true={exact_S1_W1:.3f}')  # True: 0.430

    # Use QRS with 0 and 2 iterations
    prob, acc_rate_i = QS.perform_rejection_sampling(iterations=0,
                                                     shots=n_shots,
                                                     seed=41)
    print(f'P(S=1|W=1)={prob:.3f} (true={exact_S1_W1:.3f}), acceptance ratio={acc_rate_i:.3f}')
    QS.qbn.qc.draw(output='mpl', filename='./circuit.png')

    prob, acc_rate_i = QS.perform_rejection_sampling(iterations=2,
                                                     shots=n_shots,
                                                     seed=41)
    print(f'P(S=1|W=1)={prob:.3f} (true={exact_S1_W1:.3f}), acceptance ratio={acc_rate_i:.3f}')
    return


if __name__ == "__main__":
    main()

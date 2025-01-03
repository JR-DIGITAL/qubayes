"""
This script implements the sprinkler Bayesian network network from Kevin
Murphys lecture notes: https://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html
"""
__author__ = "Florian Krebs"

import numpy as np
from qubayes.qubayes_tools import QBNQuery, Node, QBN, Graph


class QuerySprinkler(QBNQuery):

    def __init__(self):
        super(QuerySprinkler, self).__init__()
        self.target = None
        self.evidence = None
        self.graph_orig = self.create_graph()
        self.qbn = None
        self.rebuild_qbn()

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


def main():
    QS = QuerySprinkler()

    n_shots = 1024
    result = QS.qbn.perform_sampling(shots=n_shots)

    # Manual computation from the joint
    P_W1 = 0
    for k, v in result.items():
        if k[0] == '1':
            P_W1 += v
    P_W1 = P_W1 / n_shots

    P_S1W1 = 0
    for k, v in result.items():
        if k[0] == '1' and k[2] == '1':
            P_S1W1 += v
    P_S1W1 = P_S1W1 / n_shots
    print(f'P(S=1|W=1)={P_S1W1 / P_W1:.3f}, true=0.430')  # True: 0.430

    P_R1W1 = 0
    for k, v in result.items():
        if k[0] == '1' and k[1] == '1':
            P_R1W1 += v
    P_R1W1 = P_R1W1 / n_shots

    print(f'P(R=1|W=1)={P_R1W1 / P_W1:.3f}, true=0.708')  # True: 0.708

    # Use amplitude amplification
    QS.target = {'sprinkler': 'sprinkler1'}
    QS.evidence = {'wet': 'wet1'}
    prob, acc_rate_i = QS.perform_rejection_sampling(iterations=0,
                                                     shots=n_shots,
                                                     seed=41)
    print(f'P(S=1|W=1)={prob:.3f} (true=0.430), acceptance ratio={acc_rate_i:.3f}')
    QS.qbn.qc.draw(output='mpl', filename='./circuit.png')

    prob, acc_rate_i = QS.perform_rejection_sampling(iterations=2,
                                                     shots=n_shots,
                                                     seed=41)
    print(f'P(S=1|W=1)={prob:.3f} (true=0.430), acceptance ratio={acc_rate_i:.3f}')
    return


if __name__ == "__main__":
    main()

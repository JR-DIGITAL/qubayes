"""
This script implements the sprinkler Bayesian network network from Kevin
Murphys lecture notes: https://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html
"""
__author__ = "Florian Krebs"

import numpy as np
from qubayes.qubayes_tools import Query, Node, QBN, Graph


def create_graph():
    cloudy = Node('cloudy', data=np.array([0.5, 0.5]))
    sprinkler = Node('sprinkler', data=np.array([[0.5, 0.5],    # C=0
                                                 [0.9, 0.1]]),  # C=1
                     parents=['cloudy'])
    rain = Node('rain', data=np.array([[0.8, 0.2],    # C=0
                                       [0.2, 0.8]]),  # C=1
                parents=['cloudy'])
    probs_wet = np.array([[[1.0, 0.0],      # S=0, R=0
                           [0.1, 0.9]],     # S=0, R=1
                          [[0.1, 0.9],      # S=1, R=0
                           [0.01, 0.99]]])  # S=1, R=1
    wet = Node('wet', data=probs_wet,
               parents=['sprinkler', 'rain'])
    bn = Graph({'cloudy': cloudy, 'sprinkler': sprinkler, 'rain': rain, 'wet': wet})
    return bn


def main():
    bn = create_graph()

    qbn = QBN(bn)
    n_shots = 1024
    result = qbn.perform_sampling(shots=n_shots)

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
    print(f'P(S=1|W=1)={P_S1W1 / P_W1:.3f}')  # True: 0.430

    P_R1W1 = 0
    for k, v in result.items():
        if k[0] == '1' and k[1] == '1':
            P_R1W1 += v
    P_R1W1 = P_R1W1 / n_shots

    print(f'P(R=1|W=1)={P_R1W1 / P_W1:.3f}')  # True: 0.708

    # Use amplitude amplification
    class QuerySprinkler(Query):

        def __init__(self, graph):
            super(QuerySprinkler, self).__init__()
            self.target = {'sprinkler': 'sprinkler1'}
            self.evidence = {'wet': 'wet1'}
            self.graph_orig = graph
            self.qbn = QBN(graph)

    QS = QuerySprinkler(bn)
    prob, acc_rate_i = QS.perform_rejection_sampling(iterations=0,
                                                     shots=n_shots,
                                                     seed=41)
    print(f'P(S=1|W=1)={prob:.3f}, acceptance ratio={acc_rate_i:.3f}')
    QS = QuerySprinkler(bn)
    prob, acc_rate_i = QS.perform_rejection_sampling(iterations=2,
                                                     shots=n_shots,
                                                     seed=41)
    print(f'P(S=1|W=1)={prob:.3f}, acceptance ratio={acc_rate_i:.3f}')


    return


if __name__ == "__main__":
    main()
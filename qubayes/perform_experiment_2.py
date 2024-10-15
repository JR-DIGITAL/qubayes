"""
This script contains code to create Figure 4 in the paper, comparing circuit
depth and acceptance ratio of a circuit with and without amplitude
amplification.
"""
__author__ = "Florian Krebs"

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from dataset_stats import MusicDataset
from qubayes_tools import Graph, Node, QBN
from config import OUT_DIR


def create_model(nodes, n_artists, use_ancillas=False):
    bins = {'track_genre': ['rockabilly', 'blues', 'rock', 'jazz'],
            'artists': n_artists,
            'tempo': 2,
            'mode': 2, 'time_signature': 2}
    ds = MusicDataset(bins)
    graph = Graph(nodes)
    graph.set_probabilities(ds)
    graph.binarize()
    qbn = QBN(graph, use_ancillas=use_ancillas)
    evidence = {'artists': 'Ella Fitzgerald',
                'mode': 'major'}
    evidence = qbn.create_evidence_states(evidence)
    print(f'Chose {len(evidence)} states for amplification')
    return qbn, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ancillas", action='store_true', help="Use ancilla qubits for creating the C^n Ry gates.")
    parser.add_argument("--recompute", action='store_true', help="Do not use saved intermediate results.")
    parser.add_argument("--max_artists", type=int, default=512, help="Max number of artists (default: %(default)s)")
    args = parser.parse_args()
    if args.use_ancillas:
        save_fln = os.path.join(OUT_DIR, 'results_ancillas.npz')
    else:
        save_fln = os.path.join(OUT_DIR, 'results_no_ancillas.npz')

    if not os.path.exists(save_fln) or args.recompute:
        max_x = int(np.log2(args.max_artists))
        n_artists = [2**x for x in range(1, max_x + 1)]

        nodes = {'artists': Node('artists'),
                 'track_genre': Node('track_genre', parents=list(['artists'])),
                 'tempo': Node('tempo', parents=list(['track_genre'])),
                 'mode': Node('mode', parents=list(['track_genre'])),
                 'time_signature': Node('time_signature', parents=list(['track_genre']))}

        # First, perform rejection sampling without amplitude amplification.
        # This should yield the same performance as classical rejection sampling.
        depth_plain = np.zeros((len(n_artists),))
        cnots_plain = np.zeros((len(n_artists),))
        qubits_plain = np.zeros((len(n_artists),))
        acc_rate_plain = np.zeros((len(n_artists),))

        print('Without amplitude amplification')
        iterations = 0  # this corresponds to not using amplitude amplification
        for i, n_artist in enumerate(n_artists):
            print(f'Using {n_artist} artists')
            qbn, evidence = create_model(nodes, n_artist, use_ancillas=args.use_ancillas)
            result, circuit_params, acc_rate_i = qbn.perform_rejection_sampling(evidence, iterations=iterations)
            depth_plain[i] = circuit_params['depth']
            cnots_plain[i] = circuit_params['ops']['cx']
            qubits_plain[i] = qbn.n_qubits
            acc_rate_plain[i] = acc_rate_i

        print('With amplitude amplification')
        depth_aa = np.zeros((len(n_artists),))
        cnots_aa = np.zeros((len(n_artists),))
        qubits_aa = np.zeros((len(n_artists),))
        acc_rate_aa = np.zeros((len(n_artists),))
        grov_iterations = np.zeros((len(n_artists),))
        for i, n_artist in enumerate(n_artists):
            print(f'Using {n_artist} artists')
            for iterations in range(1, 4):  # we try between 1 and 3 grover iterations and use the best one
                qbn, evidence = create_model(nodes, n_artist, use_ancillas=args.use_ancillas)
                result, circuit_params, acc_rate_i = qbn.perform_rejection_sampling(evidence, iterations=iterations)
                if acc_rate_i > acc_rate_aa[i]:
                    if iterations == 1:
                        depth_aa[i] = circuit_params['depth']
                        cnots_aa[i] = circuit_params['ops']['cx']
                    qubits_aa[i] = qbn.n_qubits
                    acc_rate_aa[i] = acc_rate_i
                    grov_iterations[i] = iterations
                else:
                    break
        n_artists = np.array(n_artists)
        data = {'depth_plain': depth_plain, 'cnots_plain': cnots_plain,
                'qubits_plain': qubits_plain, 'acc_rate_plain': acc_rate_plain,
                'depth_aa': depth_aa, 'cnots_aa': cnots_aa,
                'qubits_aa': qubits_aa, 'acc_rate_aa': acc_rate_aa,
                'n_artists': n_artists, 'grov_iterations': grov_iterations}
        np.savez(save_fln, **data)
    else:
        data = np.load(os.path.join(OUT_DIR, save_fln))

    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 7))

    ax[0].plot(data['acc_rate_plain'], label='classical RS', linewidth=3, marker='x', markersize=10)
    ax[0].plot(data['acc_rate_aa'], label='quantum RS', linewidth=3, marker='o', markersize=10)
    ax[0].legend()
    ax[0].set_xticks(range(len(data['n_artists'])))
    ax[0].set_xticklabels(data['n_artists'])
    ax[0].set_ylabel('Acceptance ratio')
    ax[0].grid()

    ax[1].plot(data['depth_plain'], label='without amplitude amplification', linewidth=3, marker='x', markersize=10)
    ax[1].plot(data['depth_aa'], label='with amplitude amplification', linewidth=3, marker='o', markersize=10)
    ax[1].legend()
    ax[1].set_xticks(range(len(data['n_artists'])))
    ax[1].set_xticklabels(data['n_artists'])
    ax[1].set_ylabel('Circuit depth')
    ax[1].set_xlabel('Number of artists')
    ax[1].grid()
    plt.subplots_adjust(left=0.2)

    plt.savefig(save_fln.replace('.npz', '.png'), dpi=200)
    print(f'Saved figure to {save_fln.replace(".npz", ".png")}.')
    return


if __name__ == "__main__":
    main()

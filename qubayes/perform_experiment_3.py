"""
This script contains code to compare the complexity of classical and quantum rejection sampling.
We do this by decreasing P(evidence) step by step and measuring the number of samples needed and
the circuit depth.
"""
__author__ = "Florian Krebs"

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from dataset_stats import MusicDataset
from qubayes_tools import Graph, Node, QBN
from config import OUT_DIR
import pandas as pd

EVIDENCE = {'artists': 'Ella Fitzgerald',
            'mode': 'major'}


def create_model(nodes, p_artist, n_artists=4, use_ancillas=False, evidence=EVIDENCE):
    bins = {'track_genre': ['rockabilly', 'blues', 'rock', 'jazz'],
            'artists': n_artists,
            'tempo': 2,
            'mode': 2, 'time_signature': 2}
    ds = MusicDataset(bins)
    graph = Graph(nodes)
    graph.set_probabilities(ds)
    # adjust probability of evidence. Should be 1/n_artists
    artist_idx = graph.categories['artists'][EVIDENCE['artists']]
    mode_idx = graph.categories['mode'][EVIDENCE['mode']]
    graph.nodes['artists'].data[artist_idx] = p_artist
    # normalize the other probabilites to 1 - (1/n_artists)
    idx = np.ones((n_artists,), dtype=bool)
    idx[artist_idx] = 0
    graph.nodes['artists'].data[idx] /= graph.nodes['artists'].data[idx].sum()
    graph.nodes['artists'].data[idx] *= (1 - p_artist)
    assert(abs(graph.nodes['artists'].data.sum() - 1) < 0.0001)
    p_evidence = ((graph.nodes['artists'].data[artist_idx] *
                   graph.nodes['track_genre'].data[:, artist_idx]).T *
                  graph.nodes['mode'].data[mode_idx, :]).sum()
    assert p_evidence < 1
    graph.binarize()
    qbn = QBN(graph, use_ancillas=use_ancillas)

    evidence = qbn.create_evidence_states(evidence)
    # print(f'Chose {len(evidence)} states for amplification')
    return qbn, evidence, p_evidence


def create_plot0(data, save_fln):
    # plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 7))
    denoms = np.unique(data['denom'])
    labels = []
    for den in range(len(denoms)):
        labels.append(r'$2^{-' + str(int(den+1)) + '}$')
    p_ev = [f'{1/(2**d):.2f}' for d in denoms]
    # plot p_evidence and complexity over n_artists

    # no amplitude amplification
    df = data[(data['n_grover_it'] == 0)]

    ax[0].plot(1. / df['p_evidence'], label='Number of samples', linewidth=3, marker='x', markersize=10)
    ax[0].legend()
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel('Classical complexity')
    ax[0].grid()

    # with amplitude amplification
    complexity = np.zeros((len(denoms),))
    for i, den in enumerate(denoms):
        df = data[(data['denom'] == den) & (data['n_grover_it'] != 0)]
        acc_rate = np.array(df['acc_rate'])
        depth = np.array(df['depth'])
        comp = depth / acc_rate
        complexity[i] = comp.min()
    ax[1].plot(complexity, label='Depth / acceptance rate', linewidth=3, marker='x', markersize=10)
    ax[1].legend()
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel('Quantum complexity')
    ax[1].grid()
    ax[1].set_xlabel('P(evidence)')

    # ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fln = save_fln.replace('.csv', '_p_evidence.png')
    plt.savefig(fln, dpi=200)
    print(f'Saved figure to {fln}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ancillas", action='store_true', help="Use ancilla qubits for creating the C^n Ry gates.")
    parser.add_argument("--recompute", action='store_true', help="Recompute results.")
    parser.add_argument("--max_exponent", type=int, default=11, help="Max exponent for p_artist (default: %(default)s)")
    args = parser.parse_args()
    if args.use_ancillas:
        save_fln = os.path.join(OUT_DIR, 'results_4artists_ancillas.csv')
    else:
        save_fln = os.path.join(OUT_DIR, 'results_4artists_no_ancillas.csv')
    n_artists = 4
    if not os.path.exists(save_fln) or args.recompute:
        results = pd.DataFrame(columns=['depth', 'cnots', 'qubits',
                                        'acc_rate', 'p_evidence',
                                        'n_artists', 'n_grover_it'])
        denominators = [2**x for x in range(1, args.max_exponent + 1)]

        nodes = {'artists': Node('artists'),
                 'track_genre': Node('track_genre', parents=list(['artists'])),
                 'tempo': Node('tempo', parents=list(['track_genre'])),
                 'mode': Node('mode', parents=list(['track_genre'])),
                 'time_signature': Node('time_signature', parents=list(['track_genre']))}
        p_evidence = np.zeros((len(denominators),))

        print('Without amplitude amplification')
        iterations = 0  # this corresponds to not using amplitude amplification
        for i, denom in enumerate(denominators):
            print(f'Using {denom} denominator')
            qbn, evidence, p_evidence[i] = create_model(nodes, 1 / denom, n_artists=n_artists, use_ancillas=args.use_ancillas)
            result, circuit_params, acc_rate_i = qbn.perform_rejection_sampling(evidence, iterations=iterations)
            new_row = {'depth': circuit_params['depth'],
                       'cnots': circuit_params['ops']['cx'],
                       'qubits': qbn.n_qubits, 'acc_rate': acc_rate_i,
                       'n_grover_it': iterations, 'p_evidence': p_evidence[i],
                       'denom': denom}
            results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
            results.to_csv(save_fln, index=False)

        print('With amplitude amplification')
        for i, denom in enumerate(denominators):
            n_grover_it = int(np.ceil(np.pi / (4 * np.arcsin(np.sqrt(p_evidence[i])))))
            print(f'Using {denom}, suggested {n_grover_it} grover iterations')

            for iterations in range(1, n_grover_it):
                # we try between 1 and 3 grover iterations and use the best one
                qbn, evidence, p_evidence_it = create_model(
                    nodes, 1 / denom, n_artists=n_artists, use_ancillas=args.use_ancillas)
                result, circuit_params, acc_rate_i = qbn.perform_rejection_sampling(evidence, iterations=iterations)
                # qcomp_i = circuit_params['depth'] / acc_rate_i
                new_row = {'depth': circuit_params['depth'],
                           'cnots': circuit_params['ops']['cx'],
                           'qubits': qbn.n_qubits, 'acc_rate': acc_rate_i,
                           'n_grover_it': iterations, 'p_evidence': p_evidence_it,
                           'denom': denom}
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                results.to_csv(save_fln, index=False)
        results.to_csv(save_fln, index=False)
    else:
        results = pd.read_csv(save_fln)

    create_plot0(results, save_fln)

    return


if __name__ == "__main__":
    main()

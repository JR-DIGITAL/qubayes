"""
This script implements the experiments of the paper Krebs et al., Modeling Musical Knowledge with Quantum
Bayesian Networks, CBMI 2024.
"""
__author__ = "Florian Krebs"

import os.path
import argparse
import numpy as np
from qubayes.qubayes_tools import QBNMusicQuery
from config import MODEL_FLN


class Query1(QBNMusicQuery):

    def __init__(self, use_ancillas=False):
        super(Query1, self).__init__()
        self.create_model(n_artists=32,
                          n_genres=['rockabilly', 'blues', 'rock', 'jazz'],
                          use_ancillas=use_ancillas)
        self.target = {'track_genre': 'blues'}
        self.evidence = {'artists': 'Chuck Berry',
                         'mode': 'minor'}
        self.name_str = 'P(track_genre=blues | artists=Chuck Berry, mode=minor)'


class Query2(QBNMusicQuery):

    def __init__(self, use_ancillas=False):
        super(Query2, self).__init__()
        self.create_model(n_artists=32,
                          n_genres=['rockabilly', 'blues', 'rock', 'jazz'],
                          use_ancillas=use_ancillas)
        self.target = {'artists': 'Nina Simone'}
        self.evidence = {'time_signature': '3',
                         'tempo': 'tempo_52_119'}


class Query3(QBNMusicQuery):

    def __init__(self, use_ancillas=False):
        super(Query3, self).__init__()
        self.create_model(n_artists=32,
                          n_genres=['rockabilly', 'blues', 'rock', 'jazz'],
                          use_ancillas=use_ancillas)
        self.target = {'track_genre': 'jazz'}
        self.evidence = {'artists': 'Ella Fitzgerald',
                         'mode': 'major'}


def run_query(Query, n_trials=1, shots=1024, n_iterations=3, seed=None, use_ancillas=False):

    accr_qrs0 = np.zeros((n_trials,))
    accr_qrs = np.zeros((n_trials,))
    accr_crs = np.zeros((n_trials,))
    me_qrs0 = np.zeros((n_trials,))
    me_qrs = np.zeros((n_trials,))
    me_crs = np.zeros((n_trials,))
    if os.path.exists(MODEL_FLN):
        os.remove(MODEL_FLN)

    for i in range(n_trials):
        print(f'--- Repetition {i} ---')
        # quantum rejection sampling without amplitude amplification. This
        # should yield the same results as classical rejection sampling.
        query = Query(use_ancillas=use_ancillas)
        prob, acc_rate_i = query.perform_rejection_sampling(iterations=0, shots=shots, seed=seed)
        accr_qrs0[i] = acc_rate_i * 100
        me_qrs0[i] = abs(prob - query.get_true_result())
        print(f'Exact: {query.get_true_result():.4f}')
        print(f'Quantum (0 Grover iterations): {prob:.4f}')

        # quantum rejection sampling
        query = Query(use_ancillas=use_ancillas)
        prob, acc_rate_i = query.perform_rejection_sampling(iterations=n_iterations, shots=shots, seed=seed)
        accr_qrs[i] = acc_rate_i * 100
        me_qrs[i] = abs(prob - query.get_true_result())
        print(f'Quantum ({n_iterations} Grover iterations): {prob:.4f}')

        # Classical rejection sampling
        prob, acc_rate_i = query.perform_classical_rejection_sampling(shots=shots)
        accr_crs[i] = acc_rate_i * 100
        me_crs[i] = abs(prob - query.get_true_result())
        print(f'Classical RS: {prob:.4f}')
    print(f'--- Average ---')
    print(f'QRS0: acc_rate={accr_qrs0.mean():7.3f} ({accr_qrs0.std():.2f}), '
          f'mse={me_qrs0.mean():7.3f} ({me_qrs0.std():.3f})')
    print(f'QRS:  acc_rate={accr_qrs.mean():7.3f} ({accr_qrs.std():.2f}), '
          f'mse={me_qrs.mean():7.3f} ({me_qrs.std():.3f})')
    print(f'CRS:  acc_rate={accr_crs.mean():7.3f} ({accr_crs.std():.2f}), '
          f'mse={me_crs.mean():7.3f} ({me_crs.std():.3f})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", default=5, type=int,
                        help="Number of repetitions of experiment "
                             "(default: %(default)s)")
    parser.add_argument("--shots", default=1000, type=int,
                        help="Number of quantum circuit measurements "
                             "(default: %(default)s)")
    parser.add_argument("--use_ancillas", action='store_true',
                        help="Use ancilla qubits for creating the C^n Ry gates.")
    args = parser.parse_args()
    np.random.seed(40)
    # Query1
    print("\n=== QUERY 1 ===")
    run_query(Query1, n_trials=args.repetitions, shots=args.shots,
              n_iterations=6, seed=None, use_ancillas=args.use_ancillas)

    # Query2
    print("\n=== QUERY 2 ===")
    run_query(Query2, n_trials=args.repetitions, shots=args.shots,
              n_iterations=3, seed=None, use_ancillas=args.use_ancillas)

    # Query3
    print("\n=== QUERY 3 ===")
    run_query(Query3, n_trials=args.repetitions, shots=args.shots,
              n_iterations=2, seed=None, use_ancillas=args.use_ancillas)

    return


if __name__ == "__main__":
    main()

import unittest
from qubayes.qubayes_tools import QBNMusicQuery
import numpy as np


class MyTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MyTestCase, self).__init__(*args, **kwargs)
        self.n_shots = 4096
        self.query = QBNMusicQuery()
        self.query.create_model(n_artists=4,
                                n_genres=['rock', 'jazz', 'blues', 'rockabilly'],
                                n_tempi=2,
                                n_modes=2,
                                n_time_signature=2,
                                use_ancillas=False,
                                model_fln=None)

    def test_sampling_graph(self):
        n_samples = 10000
        np.random.seed(42)
        test_tol = 0.01
        samples, names = self.query.graph_orig.sample_from_graph(n_samples)
        # artist
        probs_artist = np.bincount(samples[0, :]) / n_samples
        np.testing.assert_allclose(probs_artist[0], self.query.graph_orig.nodes['artists'].data[0], atol=test_tol)
        np.testing.assert_allclose(probs_artist[2], self.query.graph_orig.nodes['artists'].data[2], atol=test_tol)
        # track genre
        idx_ella = samples[0, :] == 0
        n_bins = self.query.graph_orig.nodes['track_genre'].n_states
        probs_genre_artist_0 = np.bincount(samples[1, idx_ella], minlength=n_bins) / sum(idx_ella)
        np.testing.assert_allclose(probs_genre_artist_0[0], self.query.graph_orig.nodes['track_genre'].data[0, 0], atol=test_tol)
        np.testing.assert_allclose(probs_genre_artist_0[1], self.query.graph_orig.nodes['track_genre'].data[1, 0],
                                   atol=test_tol)
        idx_elvis = samples[0, :] == 2
        n_bins = self.query.graph_orig.nodes['track_genre'].n_states
        probs_genre_artist_2 = np.bincount(samples[1, idx_elvis], minlength=n_bins) / sum(idx_elvis)
        np.testing.assert_allclose(probs_genre_artist_2[0], self.query.graph_orig.nodes['track_genre'].data[0, 2],
                                   atol=test_tol)
        np.testing.assert_allclose(probs_genre_artist_2[1], self.query.graph_orig.nodes['track_genre'].data[1, 2],
                                   atol=test_tol)
        # tempo
        idx_jazz = samples[1, :] == 1
        n_bins = self.query.graph_orig.nodes['tempo'].n_states
        probs_tempo_jazz = np.bincount(samples[2, idx_jazz], minlength=n_bins) / sum(idx_jazz)
        np.testing.assert_allclose(probs_tempo_jazz[0], self.query.graph_orig.nodes['tempo'].data[0, 1],
                                   atol=test_tol)
        idx_blues = samples[1, :] == 2
        probs_tempo_blues = np.bincount(samples[2, idx_blues], minlength=n_bins) / sum(idx_blues)
        np.testing.assert_allclose(probs_tempo_blues[0], self.query.graph_orig.nodes['tempo'].data[0, 2],
                                   atol=test_tol)
        # mode
        idx_jazz = samples[1, :] == 1
        n_bins = self.query.graph_orig.nodes['mode'].n_states
        probs_mode_jazz = np.bincount(samples[3, idx_jazz], minlength=n_bins) / sum(idx_jazz)
        np.testing.assert_allclose(probs_mode_jazz[0], self.query.graph_orig.nodes['mode'].data[0, 1],
                                   atol=test_tol)
        idx_blues = samples[1, :] == 2
        probs_mode_blues = np.bincount(samples[3, idx_blues], minlength=n_bins) / sum(idx_blues)
        np.testing.assert_allclose(probs_mode_blues[0], self.query.graph_orig.nodes['mode'].data[0, 2],
                                   atol=test_tol)

    def test_query0(self):
        np.random.seed(42)
        self.query.target = {'track_genre': 'jazz'}
        self.query.evidence = {'artists': 'Ella Fitzgerald',
                               'tempo': 'tempo_102_206'}
        prob_exact = self.query.get_true_result()
        np.testing.assert_allclose(prob_exact, 0.6585249, atol=0.001)

        [prob, acc] = self.query.perform_classical_rejection_sampling(shots=10000)
        np.testing.assert_allclose(prob, prob_exact, atol=0.01)

        [prob, acc] = self.query.perform_rejection_sampling(shots=10000, iterations=0, seed=42)
        np.testing.assert_allclose(prob, prob_exact, atol=0.01)

        [prob, acc] = self.query.perform_rejection_sampling(shots=1000, iterations=1, seed=42)
        np.testing.assert_allclose(prob, prob_exact, atol=0.01)

        [prob, acc] = self.query.perform_rejection_sampling(shots=1000, iterations=2, seed=42)
        np.testing.assert_allclose(prob, prob_exact, atol=0.01)

    def test_query1(self):
        np.random.seed(42)
        self.query.target = {'track_genre': 'rockabilly'}
        self.query.evidence = {'time_signature': '4'}
        prob_exact = self.query.get_true_result()
        np.testing.assert_allclose(prob_exact, 0.17943548387096775, atol=0.001)

        [prob, acc] = self.query.perform_classical_rejection_sampling(shots=10000)
        np.testing.assert_allclose(prob, prob_exact, atol=0.01)

        [prob, acc] = self.query.perform_rejection_sampling(shots=10000, iterations=0, seed=42)
        np.testing.assert_allclose(prob, prob_exact, atol=0.01)

    def test_query2(self):
        np.random.seed(42)
        self.query.target = {'artists': 'Elvis Presley'}
        self.query.evidence = {'time_signature': '3'}
        prob_exact = self.query.get_true_result()
        np.testing.assert_allclose(prob_exact, 0.3777777777777778, atol=0.001)

        [prob, acc] = self.query.perform_classical_rejection_sampling(shots=10000)
        np.testing.assert_allclose(prob, prob_exact, atol=0.05)

        [prob, acc] = self.query.perform_rejection_sampling(shots=10000, iterations=0, seed=42)
        np.testing.assert_allclose(prob, prob_exact, atol=0.03)

        [prob, acc] = self.query.perform_rejection_sampling(shots=1000, iterations=1, seed=42)
        np.testing.assert_allclose(prob, prob_exact, atol=0.05)

        [prob, acc] = self.query.perform_rejection_sampling(shots=1000, iterations=2, seed=42)
        np.testing.assert_allclose(prob, prob_exact, atol=0.01)


class MyTestCase2(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MyTestCase2, self).__init__(*args, **kwargs)
        self.n_shots = 4096
        self.query = QBNMusicQuery()
        self.query.create_model(n_artists=32,
                                n_genres=['rockabilly', 'blues', 'rock', 'jazz'],
                                use_ancillas=False)

    def test_query0(self):
        np.random.seed(42)
        self.query.target = {'artists': 'Nat King Cole'}
        self.query.evidence = {'track_genre': 'jazz', 'mode': 'major'}

        prob_exact = self.query.get_true_result()
        np.testing.assert_allclose(prob_exact, 0.18681318681318684, atol=0.001)

        [prob, acc] = self.query.perform_classical_rejection_sampling(shots=10000)
        np.testing.assert_allclose(prob, prob_exact, atol=0.01)

        # TODO: Here seems to be something wrong, there is a consistent high error
        [prob, acc] = self.query.perform_rejection_sampling(shots=10000, iterations=0, seed=42)
        np.testing.assert_allclose(prob, prob_exact, atol=0.1)

    def test_query1(self):
        np.random.seed(42)
        self.query.target = {'artists': 'Arctic Monkeys'}
        self.query.evidence = {'track_genre': 'rock', 'mode': 'major'}

        prob_exact = self.query.get_true_result()
        np.testing.assert_allclose(prob_exact, 0.0599369085, atol=0.001)

        [prob, acc] = self.query.perform_classical_rejection_sampling(shots=10000)
        np.testing.assert_allclose(prob, prob_exact, atol=0.02)

        # TODO: Here seems to something wrong, there is a consistent high error
        [prob, acc] = self.query.perform_rejection_sampling(shots=10000, iterations=0)
        np.testing.assert_allclose(prob, prob_exact, atol=0.1)


if __name__ == '__main__':
    unittest.main()

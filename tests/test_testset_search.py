"""离线搜索的批处理、坐标迭代及历史编码复用。"""

import unittest

import numpy as np
import torch

from baseline.compare_testset_search import FrozenStateScorer, search_batch
from baseline.lstm import GridLSTM, LSTMPredictor
from baseline.sequence import CONTEXT_COLUMNS, SequencePreprocessor
from tests.test_lstm import HISTORY_FEATURES, _dataset


class MatrixScorer:
    grid_count = 2
    candidate_count = 2

    def score(self, encoded, context, actions):
        return np.array([[1., 4.], [2., 3.]])[actions[..., 0], actions[..., 1]]


class TestSetSearchTests(unittest.TestCase):
    def test_bcs_revisits_coordinates_and_stops_each_state_independently(self):
        initial = np.array([[0, 0], [0, 1]])
        dummy = torch.zeros(2, 1)
        greedy, bcs, stable, sweeps, *_ = search_batch(MatrixScorer(), dummy, dummy, initial)
        np.testing.assert_array_equal(greedy, [[1, 1], [0, 1]])
        np.testing.assert_array_equal(bcs, [[0, 1], [0, 1]])
        np.testing.assert_array_equal(stable, [True, True])
        np.testing.assert_array_equal(sweeps, [3, 1])
        np.testing.assert_array_equal(initial, [[0, 0], [0, 1]])

    def test_single_sweep_matches_greedy_and_constant_scores_keep_current(self):
        initial = np.array([[0, 0], [0, 1]])
        dummy = torch.zeros(2, 1)
        greedy, bcs, stable, sweeps, *_ = search_batch(MatrixScorer(), dummy, dummy, initial, 1)
        np.testing.assert_array_equal(greedy, bcs)
        np.testing.assert_array_equal(stable, [False, True])
        np.testing.assert_array_equal(sweeps, [1, 1])

        class TiedScorer(MatrixScorer):
            def score(self, encoded, context, actions):
                return np.ones(actions.shape[:2])

        _, bcs, stable, sweeps, *_ = search_batch(TiedScorer(), dummy, dummy, initial)
        np.testing.assert_array_equal(bcs, initial)
        self.assertTrue(stable.all())
        self.assertTrue((sweeps == 1).all())

    def test_reused_history_encoding_matches_full_model_forward(self):
        frame = _dataset()
        extra = []
        for grid in (7, 31):
            for field, value in (("price", 1.3), ("radius", 2.0)):
                name = f"joint_{field}_grid_{grid}"
                frame[name] = value
                extra.append(name)
        processor = SequencePreprocessor(2, HISTORY_FEATURES, list(CONTEXT_COLUMNS) + extra)
        processor.fit(frame.loc[frame.split == "train"])
        model = GridLSTM(len(HISTORY_FEATURES), 2, hidden_size=6, embedding_size=3,
                         dropout=0, context_size=len(processor.context_columns))
        predictor = LSTMPredictor(model, processor, "cpu")
        scorer = FrozenStateScorer(predictor, [1., 1.3], [1., 2.])
        h, c, g = processor.transform_inputs(frame.iloc[:4])
        history, context, grids = h.reshape(2, 2, 2, -1), torch.from_numpy(c.reshape(2, 2, -1)), g.reshape(2, 2)
        actions = np.array([[[0, 1], [2, 3], [1, 2]]] * 2)
        with torch.inference_mode():
            encoded = scorer.encode(history, grids)
            score = scorer.score(encoded, context, actions)
            candidate_context = scorer.contexts(context, actions)
            repeated_history = torch.from_numpy(np.repeat(history[:, None], 3, axis=1)).reshape(12, 2, -1)
            repeated_grids = torch.from_numpy(np.repeat(grids[:, None], 3, axis=1)).reshape(-1)
            reference = model(repeated_history, candidate_context.reshape(12, -1), repeated_grids)
        values = processor.inverse_targets(reference.numpy()).reshape(2, 3, 2, 6)[..., 5]
        np.testing.assert_allclose(score, values.sum(axis=2, dtype=np.float64), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
